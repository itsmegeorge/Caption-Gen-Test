# Caption-Gen-Test
Jason Brownlee tutorial, some changes. Need to revisit.
It is trained using Flickr8K Dataset and VGG16. A more ideal
setting involves using MSCOCO instead and Google's InceptionV3.
Add more features to repeat vectors also, but then training takes longer.
If you want to build the model, you have to download Flickr8K and put everything in the same foler. Dividing dataset not necessary, done automatically. Put the pictures in the same dir and run python3 ./extract_features.py. Then run python3 ./build_model.py and when it's done training, do python3 ./tokenizer.py. Put an example.jpg image in the dir and run python3 ./caption_generator.py
