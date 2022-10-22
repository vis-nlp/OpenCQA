import argparse
import logging
from tqdm import trange
import torch
import torch.nn.functional as F
import numpy as np
from torch import nn
from torch.autograd import Variable
# from transformers import GPT2Config
# from transformers import GPT2LMHeadModel, GPT2Tokenizer, BertTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM
from DataLoader import *
# from Model import BERTGen
from NucleusSampling import sample_sequence
from ChartLayer import ChartLayerModule
import torch.optim as optim
import math
import sys
import pandas
import os
import numpy
import nltk
import re
import time
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

device = torch.device('cpu')
torch.cuda.empty_cache()

def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def clean_str(strings):
    new_strings = []
    for string in strings:
        string = re.sub(r' +', ' ', string)
        if len(string.split(' ')) < 6 and len(new_strings) > 0:
            string = new_strings[-1]
        new_strings.append(string)
    return new_strings

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='gptneo', type=str)
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="temperature of 0 implies greedy sampling")
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--top_p", type=float, default=0.9)
    # parser.add_argument('--seed', type=int, default=42,
    #                     help="random seed for initialization")
    parser.add_argument('--do_train', default=True, action="store_true",
                        help="whether to train or test the model")
    parser.add_argument('--do_test', default=False, action="store_true",
                        help="whether to train or test the model")
    parser.add_argument('--do_verify', default=False, action="store_true",
                        help="whether to train or test the model")
    parser.add_argument('--epoch', default=10, type=int,
                        help="whether to train or test the model")
    parser.add_argument('--pretuning_epoch', default=5, type=int,
                        help="whether to train or test the model")
    parser.add_argument('--batch_size', default=6, type=int,
                        help="whether to train or test the model")
    # parser.add_argument('--do_test_challenge', default=False, action="store_true", 
    #                     help="whether to train or test the model")
    parser.add_argument('--transfer_learn', default=False, action="store_true", 
                        help="transfer learn a new layer for the task")
    parser.add_argument('--tl_learning_rate', default=2e-4, type=float, 
                        help="transfer learn learning rate")
    parser.add_argument('--ft_learning_rate', default=2e-6, type=float, 
                        help="fine tune learning rate")
    parser.add_argument('--chart_load_format', default='bbox', type=str, 
                        help="whether to train or test the model")
    # parser.add_argument('--do_verify_challenge', default=False, action="store_true", 
    #                     help="whether compute the adv-acc score on challenge split")    
    parser.add_argument('--every', default=50, type=int, help="evaluate how many steps")
    parser.add_argument('--start_epoch', default=0, type=int, help="resuming from which epoch")
    parser.add_argument('--load_from', default='checkpoints/GPTNEO_ep9.pt', type=str, help="load model path")
    parser.add_argument('--load_bboxes_from', default='data/bboxes/', type=str, help="load bboxes path")
    parser.add_argument('--load_charts_from', default='data/chart_images/', type=str, help="load charts path")
    parser.add_argument('--id', default='checkpoints', type=str, help="specify the id of the experiment")
    parser.add_argument('--max_len', default=500, type=int, help="max seq length model can take is 2048")
    # parser.add_argument('--stage', default=1, type=int, help="whether to train or test the model")
    # parser.add_argument("--modelpath", type=str, default="bert-base-uncased",
    #                     help="For distributed training: local_rank")
    # parser.add_argument('--gradient_accumulation_steps', type=int, default=5, help="accumulation steps for gradient")
    parser.add_argument('--decode_first_K', type=int, default=10000, help="For debugging purpose")    
    args = parser.parse_args()

    args.device = torch.device("cpu")
    args.n_gpu = torch.cuda.device_count()
    # start_epoch = args.start_epoch
    # if args.model == 'gpt2-medium':
    #     args.batch_size = 2

    print(args)

    # tokenizer = GPT2Tokenizer.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")

    model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-neo-125M")
    
    if args.transfer_learn == True:
        ext_model = ChartLayerModule(model)
        print("transferring model to selected device: ",args.device)
        ext_model = nn.DataParallel(ext_model)
        ext_model.to(args.device)
    else:
        print("transferring model to selected device: ",args.device)
        # model = nn.DataParallel(model)
        # model.to(args.device)
    # tokenizer.add_tokens(['[ENT]', '[SEP]'])

    # model = GPT2LMHeadModel.from_pretrained(args.model)
    # model.resize_token_embeddings(len(tokenizer))
    
    
    chart_dim = 2560 #temp till krl of charts are available
    
    if not os.path.exists(args.id):
        os.mkdir(args.id)

    criterion = nn.CrossEntropyLoss(reduction='none', ignore_index=-1)
    model.train()
    if args.do_train:
        recording_time = datetime.now().strftime('%m_%d_%H_%M')
        tb_writer = SummaryWriter(log_dir='tensorboard/GPTNEO_ep/{}'.format(recording_time))        
        dataset = GPTNeoLoadData('data/train.json', None, None,tokenizer, args.chart_load_format, args.load_bboxes_from, args.batch_size)
        # if args.stage == 2:
        #     model.load_state_dict(torch.load(args.load_from))

        
        

        avg_loss = 0
        global_step = 0      
        
        
        if args.transfer_learn == True:
                    # model = nn.Sequential(model,nn.Linear(50257, 50257,bias=False))
                    # ext_model = ChartLayerModule(model)
                    # logits = model(chart_embeddings,inputs)[0]
                    ext_model.eval() #set to inference mode for batchnorm and dropout
                    for param in model.parameters():
                        param.requires_grad = False
                    
                    ext_model.chart_embed.weight.requires_grad = True #only learn new chart layer
                    optimizer = optim.Adam(model.parameters(), args.tl_learning_rate)
#start transfer learning             
                    for epoch_idx in range(args.pretuning_epoch):
                        print("start transfer learning {}th epoch".format(epoch_idx))
                        for idx in range(0, dataset.train_len()):
                            batch = dataset.get_train_data(idx)
                            batch = tuple(Variable(t).to(device) for t in batch)
                            decoder_inputs, decoder_outputs, masks, encoder_input = batch
                            inputs = torch.cat([encoder_input, decoder_inputs], 1)
                            inputs = inputs[:,:args.max_len]
            
                            model.zero_grad()
                            optimizer.zero_grad()
                            if args.chart_load_format == 'krl':
                                chart_embeddings = dataset.extract_krl(*list(inputs.shape),chart_dim) #temp chart embeddings
                                input_embeddings = ext_model.base_model.transformer.wte(inputs)
                                logits = ext_model(inputs_embeds = torch.cat([chart_embeddings,input_embeddings], 1))
                            else:
                                input_embeddings = ext_model.base_model.transformer.wte(inputs)
                                logits = ext_model(inputs_embeds = input_embeddings)
        

                            logits = logits[:, -decoder_outputs.shape[1]:, :].contiguous()
                        
        
                            loss = criterion(logits.view(-1, logits.shape[-1]), decoder_outputs.view(-1))
            
                            loss = loss * masks.view(-1)
                            loss = loss.sum() / masks.sum()
            
                            avg_loss += loss.item()
            
                            loss.backward()
                            optimizer.step()
                            global_step += 1
                            
                            if idx % args.every == 0 and idx > 0:
                                 tb_writer.add_scalar("perplexity", math.exp(avg_loss / args.every), global_step)
                                 avg_loss = 0
  

                
                    model.zero_grad()
                    optimizer.zero_grad()
                    
                    #reset gradients to True but keep in eval mode
                    for param in model.parameters():
                        param.requires_grad = True
                    
                    #maybe put droput layers back in train mode
  
                    torch.save(model.state_dict(), '{}/GPTNEO_ep_-1.pt'.format(args.id))
#start fine tuning
        optimizer = optim.Adam(model.parameters(), args.ft_learning_rate)
        
        
        
        
        for epoch_idx in range(args.epoch):
            print("start fine tuning {}th epoch".format(epoch_idx))
            for idx in range(0, dataset.train_len()):
                batch = dataset.get_train_data(idx)
                batch = tuple(Variable(t).to(device) for t in batch)
                decoder_inputs, decoder_outputs, masks, encoder_input = batch
                inputs = torch.cat([encoder_input, decoder_inputs], 1)
                inputs = inputs[:,:args.max_len]

                model.zero_grad()
                optimizer.zero_grad()
                
                
                if args.chart_load_format == 'krl':
                    chart_embeddings = dataset.extract_krl(*list(inputs.shape),chart_dim) #temp chart embeddings
                    if args.transfer_learn == True:
                        input_embeddings = ext_model.base_model.transformer.wte(inputs)
                        logits = ext_model(inputs_embeds = torch.cat([chart_embeddings,input_embeddings], 1))
                    else:
                        input_embeddings = model.transformer.wte(inputs)
                        logits = model(inputs_embeds = torch.cat([chart_embeddings,input_embeddings], 1))[0]
                else:
                    if args.transfer_learn == True:
                        input_embeddings = ext_model.base_model.transformer.wte(inputs)
                        logits = ext_model(inputs_embeds = input_embeddings)
                    else:
                        logits = model(inputs)[0]
                   
                    
                   
                logits = logits[:, -decoder_outputs.shape[1]:, :].contiguous()
                

                loss = criterion(logits.view(-1, logits.shape[-1]), decoder_outputs.view(-1))

                loss = loss * masks.view(-1)
                loss = loss.sum() / masks.sum()

                avg_loss += loss.item()

                loss.backward()
                optimizer.step()
                global_step += 1

                if idx % args.every == 0 and idx > 0:
                    tb_writer.add_scalar("perplexity", math.exp(avg_loss / args.every), global_step)

                    # fake_inputs = caption
                    # gt_inputs = trg_out.cpu().data.numpy()

                    # samples = sample_sequence(model, 50, fake_inputs, [])
                    # samples = samples[:, caption.shape[1]:]
                    # samples = samples.cpu().data.numpy()

                    # for s, gt in zip(samples, gt_inputs):
                    #     print("EPOCH {}; FINISHED {}/{}".format(epoch_idx, idx, dataset.train_len()))
                    #     text = tokenizer.decode(s, clean_up_tokenization_spaces=True)
                    #     text = text[: text.find(tokenizer.eos_token)]
                    #     print("PREDICTION |||||| ", text)
                    #     text = tokenizer.decode(gt, clean_up_tokenization_spaces=True)
                    #     text = text[: text.find(tokenizer.eos_token)]
                    #     print("GROUNDTRUH |||||| ",text)
                    #     break

                    avg_loss = 0

            #if args.model == 'gpt2':
            torch.save(model.state_dict(), '{}/GPTNEO_ep{}.pt'.format(args.id, epoch_idx))
            # else:
            #     torch.save(model.state_dict(), '{}/GPT_stage{}_C2F_medium_ep{}.pt'.format(args.id, args.stage, epoch_idx))

    if args.do_test:
        # assert 'stage2' in args.load_from, "The testing can only be done with stage2 model"
        dataset = GPTNeoLoadData(None, None, 'data/test.json',tokenizer, args.chart_load_format, args.load_bboxes_from, args.batch_size)
        if args.transfer_learn == True:
            ext_model.load_state_dict(torch.load(args.load_from))
            chosen_model = ext_model
            
        else:
            model.load_state_dict(torch.load(args.load_from))
            chosen_model = model
            

        chosen_model.eval()        

        sent_bleus_1 = []
        sent_bleus_2 = []
        sent_bleus_3 = []

        results = {}
        start_time = time.time()
        with torch.no_grad():
            for idx in range(0, min(args.decode_first_K, dataset.test_len())):
                # batch = dataset.get_data(idx, 'test')
                batch = dataset.get_test_data(idx)
                references = dataset.get_reference(idx, 'test')
                sample_id = dataset.get_table_id(idx, 'test')
                results[sample_id] = []

                batch = tuple(Variable(t).to(device) for t in batch)
                decoder_inputs, decoder_outputs, masks, encoder_input = batch
                encoder_input = encoder_input[:,:args.max_len]
                
                
                
                if args.chart_load_format == 'krl':
                    chart_embeddings = dataset.extract_krl(*list(encoder_input.shape),chart_dim) #temp chart embeddings
                    if args.transfer_learn == True:
                        input_embeddings = chosen_model.base_model.transformer.wte(encoder_input)
                        fake_inputs = torch.cat([chart_embeddings,input_embeddings], 1)
                        # logits = ext_model(inputs_embeds = torch.cat([chart_embeddings,input_embeddings], 1))
                    else:
                        input_embeddings = chosen_model.transformer.wte(encoder_input)
                        fake_inputs = torch.cat([chart_embeddings,input_embeddings], 1)
                        # logits = model(inputs_embeds = torch.cat([chart_embeddings,input_embeddings], 1))[0]
                else:
                    if args.transfer_learn == True:
                        input_embeddings = chosen_model.base_model.transformer.wte(encoder_input)
                        fake_inputs = input_embeddings
                        # logits = ext_model(inputs_embeds = input_embeddings)
                    else:
                        fake_inputs = encoder_input
                        # logits = model(inputs)[0]
                
                
                
                
                
                
                
                # fake_inputs = encoder_input

                samples = sample_sequence(chosen_model, 100, fake_inputs, args.transfer_learn, args.chart_load_format, stop_token=tokenizer.eos_token_id, 
                                          top_k=1, trigger=None, 
                                          supress=None,
                                          repetition=None)

                samples = samples[:, encoder_input.shape[1]:]
                samples = samples.cpu().data.numpy()

                intermediate = []
                for s in samples:
                    text = tokenizer.decode(s, clean_up_tokenization_spaces=True)
                    # text = text[text.find('[SEP]') + 6: text.find(tokenizer.eos_token)].strip()
                    text = text[: text.find(tokenizer.eos_token)].strip()
                    intermediate.append(text)

                results[sample_id] = clean_str(intermediate)

                for text in results[sample_id]:
                    hypothesis = text.lower().split()
                    sent_bleus_1.append(nltk.translate.bleu_score.sentence_bleu(
                        references, hypothesis, weights=(1, 0, 0)))
                    sent_bleus_2.append(nltk.translate.bleu_score.sentence_bleu(
                        references, hypothesis, weights=(0.5, 0.5, 0)))
                    sent_bleus_3.append(nltk.translate.bleu_score.sentence_bleu(
                        references, hypothesis, weights=(0.33, 0.33, 0.33)))

                bleu_1 = format((sum(sent_bleus_1) / len(sent_bleus_1) * 100), '.2f')
                bleu_2 = format((sum(sent_bleus_2) / len(sent_bleus_2) * 100), '.2f')
                bleu_3 = format((sum(sent_bleus_3) / len(sent_bleus_3) * 100), '.2f')

                sys.stdout.write("finished {}/{}; BLEU score {}/{}/{}; speed={}s/sent \r".format(idx, 
                                  dataset.test_len(), bleu_1, bleu_2, bleu_3, (time.time() - start_time) / len(sent_bleus_1)))

            print("total corpus BLEU score = {}/{}/{}".format(bleu_1, bleu_2, bleu_3))

        with open('outputs/{}_{}.json'.format(args.model, bleu_3), 'w') as f:
            json.dump(results, f, indent=2)

    # if args.do_test_challenge:
    #     assert 'stage2' in args.load_from, "The testing can only be done with stage2 model"
    #     dataset = GPTTableCoarseFineDatabase2(None, None, 'challenge/blind_test_lm_inputs.json', tokenizer, args.batch_size, args.max_len, args.stage)
    #     model.load_state_dict(torch.load(args.load_from))
    #     model.eval()

    #     sent_bleus_1 = []
    #     sent_bleus_2 = []
    #     sent_bleus_3 = []

    #     results = {}
    #     start_time = time.time()
    #     with torch.no_grad():
    #         for idx in range(0, min(args.decode_first_K, dataset.test_len())):
    #             batch = dataset.get_data(idx, 'test')
    #             references = dataset.get_reference(idx, 'test')
    #             table_id = dataset.get_table_id(idx, 'test')
    #             results[table_id] = []

    #             batch = tuple(Variable(t).to(device) for t in batch)
    #             trg_inp, trg_out, mask, caption = batch

    #             fake_inputs = caption

    #             samples = sample_sequence(model, 50, fake_inputs, [], stop_token=tokenizer.eos_token_id, 
    #                                      top_k=1, trigger=tokenizer.convert_tokens_to_ids('[SEP]'), 
    #                                      supress=[tokenizer.convert_tokens_to_ids('[SEP]'), 
    #                                               tokenizer.convert_tokens_to_ids('[ENT]')],
    #                                      repetition=tokenizer.convert_tokens_to_ids('[ENT]'))

    #             samples = samples[:, caption.shape[1]:]
    #             samples = samples.cpu().data.numpy()

    #             intermediate = []
    #             for s in samples:
    #                 text = tokenizer.decode(s, clean_up_tokenization_spaces=True)
    #                 text = text[text.find('[SEP]') + 6: text.find(tokenizer.eos_token)].strip()
    #                 #text = text[: text.find(tokenizer.eos_token)]
    #                 intermediate.append(text)

    #             results[table_id] = clean_str(intermediate)

    #             sys.stdout.write("finished {}/{}; speed={}s/sent \r".format(idx, 
    #                              dataset.test_len(), (time.time() - start_time) / len(results)))

    #     with open('challenge/test_results.json', 'w') as f:
    #         json.dump(results, f, indent=2)

    # if args.do_verify:
    #     assert 'stage2' in args.load_from, "The testing can only be done with stage2 model"
    #     assert args.stage == 2, "The verification can only be done with stage 2 model"
    #     dataset = GPTTableCoarseFineDatabase2(None, None, 'data/test_lm_pos_neg.json', 
    #                                          tokenizer, args.batch_size, args.max_len, args.stage)

    #     model.load_state_dict(torch.load(args.load_from))
    #     model.eval()
    #     correct, total = 0, 0
    #     with torch.no_grad():
    #         for idx in range(0, dataset.test_len()):
    #             batch_pos, batch_neg = dataset.get_pair_data(idx, 'test', mask_type='both')

    #             batch = tuple(Variable(t).to(device) for t in batch_pos)
    #             trg_inp, trg_out, mask, caption = batch

    #             inputs = torch.cat([caption, trg_inp], 1)

    #             logits = model(inputs)[0][:, -trg_out.shape[1]:, :].contiguous()

    #             loss = criterion(logits.view(-1, logits.shape[-1]), trg_out.view(-1))
    #             pos_loss = loss.reshape(logits.shape[0], -1) * mask
    #             pos_loss_per_instance = pos_loss.sum(1) / mask.sum(1)

    #             batch = tuple(Variable(t).to(device) for t in batch_neg)
    #             trg_inp, trg_out, mask, caption = batch

    #             inputs = torch.cat([caption, trg_inp], 1)

    #             logits = model(inputs)[0][:, -trg_out.shape[1]:, :].contiguous()

    #             loss = criterion(logits.view(-1, logits.shape[-1]), trg_out.view(-1))
    #             neg_loss = loss.reshape(logits.shape[0], -1) * mask
    #             neg_loss_per_instance = neg_loss.sum(1) / mask.sum(1)

    #             comparison = (pos_loss_per_instance < neg_loss_per_instance).float()
    #             correct += comparison.sum(-1).item()
    #             total += comparison.shape[0]
    #             sys.stdout.write('finished {}/{} accuracy {} \r'.format(idx, dataset.test_len(), correct / total))
    #     print('total accuracy = {}'.format(correct / total))

    # if args.do_verify_challenge:
    #     assert 'stage2' in args.load_from, "The testing can only be done with stage2 model"
    #     assert args.stage == 2, "The verification can only be done with stage 2 model"
    #     dataset = GPTTableCoarseFineDatabase2(None, None, 'challenge/blind_test_lm_pos_neg.json', 
    #                                          tokenizer, args.batch_size, args.max_len, args.stage)

    #     model.load_state_dict(torch.load(args.load_from))
    #     model.eval()
    #     correct, total = 0, 0
    #     results = {}
    #     with torch.no_grad():
    #         for idx in range(0, dataset.test_len()):
    #             batch_pos, batch_neg = dataset.get_pair_data(idx, 'test', mask_type='both')
                
    #             table_name = dataset.get_item(idx, 'test')
    #             results[table_name] = []

    #             batch = tuple(Variable(t).to(device) for t in batch_pos)
    #             trg_inp, trg_out, mask, caption = batch

    #             inputs = torch.cat([caption, trg_inp], 1)

    #             logits = model(inputs)[0][:, -trg_out.shape[1]:, :].contiguous()

    #             loss = criterion(logits.view(-1, logits.shape[-1]), trg_out.view(-1))
    #             pos_loss = loss.reshape(logits.shape[0], -1) * mask
    #             pos_loss_per_instance = pos_loss.sum(1) / mask.sum(1)
    #             pos_perpelexity_per_instance = torch.exp(pos_loss_per_instance.cpu().data).tolist()

    #             batch = tuple(Variable(t).to(device) for t in batch_neg)
    #             trg_inp, trg_out, mask, caption = batch

    #             inputs = torch.cat([caption, trg_inp], 1)

    #             logits = model(inputs)[0][:, -trg_out.shape[1]:, :].contiguous()

    #             loss = criterion(logits.view(-1, logits.shape[-1]), trg_out.view(-1))
    #             neg_loss = loss.reshape(logits.shape[0], -1) * mask
    #             neg_loss_per_instance = neg_loss.sum(1) / mask.sum(1)
    #             neg_perpelexity_per_instance = torch.exp(neg_loss_per_instance.cpu().data).tolist()

    #             for p1, p2 in zip(pos_perpelexity_per_instance, neg_perpelexity_per_instance):
    #                 if p1 < p2:
    #                     results[table_name].append('unknown1')
    #                 else:
    #                     results[table_name].append('unknown2')

    #             sys.stdout.write('finished {}/{}\r'.format(idx, dataset.test_len()))
        
    #     with open('challenge/verify_results.json', 'w') as f:
    #         json.dump(results, f, indent=2)
