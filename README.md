# anime image checker
a terminal based image checker from huggingface repo of [anime_classification](https://huggingface.co/deepghs/anime_classification  "anime_classification"). The difference from huggingface is that the project can be run in offline mode. this project also not have any interface from the original because i dont like it.

see [qna](#q&a).

## how to install
first clone this repo by using :
```
git clone https://github.com/ripaaf/anime-image-checker.git
```
then you can download all the `requirements.txt` for this able to run
```
pip install -r requirements.txt
```
you can run the project from the `main.py` file.
### downloading all the model
you can simply enambling the code in `classify.py` to download all the models. the line you looking for is line 18 :

*remove the hastag infront of the code*
```
download_all_models(repository)
```
then if all model already downloaded you can simply disabled it again by putting hastag infront the code. 
> why disable it? because it more faster to skip the process of downloading all the model the verify it rather then it checks all the downloaded model.

### explanation usage
all the code for modifying are in the `main.py` 
```
if __name__ == "__main__":
    task_name = "Is That Anime?"  
    model_name = None 
    imgsize = 384 

    image_path = "images.jpg"

    image_folder = "your/image/path/here"
    output_folder = "your/output/path/here"

    run_classification(task_name, image_path, model_name, imgsize)

    # classify_images_in_folder(task_name, image_folder, output_folder, model_name, imgsize)
```
explanation about all the variable :

`task_name` is the task you want to use, like Classification, Monochrome, AI Check, etc.

`model_name` Set a specific model if desired, or leave None to use the default

> all the models you can use is can be seen in the `repo_map.py` or in [model accuracy](#models-accuracy).

`imgsize` the size of the image before getting pass into the classification.

`image_path` is used when you want to test only for 1 image at the time.

`image_folder` is used when you want to classify multiple image, put your folder contains all the image path here.

`output_folder` the output of the multiple image classifying, the output itself making a subfolder inside where the image have been classified into.

the activate the function of 1 image only you can enable :
```
run_classification(task_name, image_path, model_name, imgsize)
```
and then for multiple image you can enable :
```
classify_images_in_folder(task_name, image_folder, output_folder, model_name, imgsize)
```

## Q&A
#### where is the model stored?
all the model is stored in the same directory of the file.

#### why no interface?
because it's complicated so I removed it so that the code can run more efficiently and quickly
#### what it used to classified the image?
it used AI that already pre-trained for only this spesific case. the model for the AI is small but it have many models. all the models if downloaded it can take up to 10gb of memory used.
#### why you do this? why dont use the 1 image only at time?
because i need this to organize my gallery😭, my gallery is so ruin that i have to make a spesific app to organize it. also i modified this for processing multiple anime photo only. there are other photo organizer i made but i later publish it when the code is not messy.

## models accuracy
all the models accuracy  you can see on the models page.
- [Classification](https://huggingface.co/deepghs/anime_classification)
- [Monochrome](https://huggingface.co/deepghs/monochrome_detect)
- [Completeness](https://huggingface.co/deepghs/anime_completeness)
- [AI Check](https://huggingface.co/deepghs/anime_ai_check)
- [AI Corrupt](https://huggingface.co/deepghs/ai_image_corrupted)
- [Rating](https://huggingface.co/deepghs/anime_rating)
- [Character Sex](https://huggingface.co/deepghs/anime_ch_sex)
- [Portrait Type](https://huggingface.co/deepghs/anime_portrait)
- [Age of Style](https://huggingface.co/deepghs/anime_style_ages)
- [Bangumi Portrait](https://huggingface.co/deepghs/bangumi_char_type)
- [Is That Anime?](https://huggingface.co/deepghs/anime_real_cls)
- [Teen](https://huggingface.co/deepghs/anime_teen)
- [Character Skin](https://huggingface.co/deepghs/anime_ch_skin_color)
- [Character Hair Color](https://huggingface.co/deepghs/anime_ch_hair_color)
- [Character Eye Color](https://huggingface.co/deepghs/anime_ch_eye_color)
- [Character Hair Length](https://huggingface.co/deepghs/anime_ch_hair_length)
- [Character Ears](https://huggingface.co/deepghs/anime_ch_ear)
- [Character Horns](https://huggingface.co/deepghs/anime_ch_horn)
- [[Beta] Danbooru Rating](https://huggingface.co/deepghs/anime_dbrating)
- [[Beta] Danbooru Aesthetic](https://huggingface.co/deepghs/anime_aesthetic)


