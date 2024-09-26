# reference from https://huggingface.co/deepghs/anime_classification 

from classify import run_classification, classify_images_in_folder

# usage
if __name__ == "__main__":
    task_name = "Is That Anime?"
    model_name = None
    imgsize = 384

    image_path = "images.jpg"

    image_folder = "your_path_here"
    output_folder = "your_path_here"

    # for 1 image only #
    run_classification(task_name, image_path, model_name, imgsize)

    # for multiple image #
    # classify_images_in_folder(task_name, image_folder, output_folder, model_name, imgsize)