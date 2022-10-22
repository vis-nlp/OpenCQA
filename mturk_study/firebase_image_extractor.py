# -*- coding: utf-8 -*-




from google.cloud import storage


client = storage.Client.from_service_account_json('descriptive-qa-firebase-adminsdk-1v7kw-c914f12e63.json')

# client = storage.Client()
bucket = client.get_bucket('descriptive-qa.appspot.com')
blob = bucket.blob('1.png')
# blob.upload_from_string('this is test content!')
down_link = blob._get_download_url
blob._do_download()

