# reference from https://huggingface.co/deepghs/anime_classification 

from classify import run_classification, classify_images_in_folder

# usage
if __name__ == "__main__":
    task_name = "Is That Anime?"  # Choose your task: Classification, Monochrome, AI Check, etc.
    model_name = None  # Set a specific model if desired, or leave None to use the default
    imgsize = 384  # You can adjust the image size for inference

    image_path = "images.jpg"  # for 1 photo only

    image_folder = "D:/Ripa/sorting whatsapp/hape/whatsap/arsip 23 juni 2023/foto"  # Provide the path to the folder containing images
    output_folder = "D:/Ripa/sorting whatsapp/hape/whatsap/arsip 23 juni 2023/foto/sorted"  # Specify the output folder

    # for 1 image only #
    run_classification(task_name, image_path, model_name, imgsize)

    # for multiple image #
    # classify_images_in_folder(task_name, image_folder, output_folder, model_name, imgsize)