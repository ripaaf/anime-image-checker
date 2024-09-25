import os
from PIL import Image
from download_models import download_all_models, print_results_in_percentage, _img_encode
from huggingface_hub import HfFileSystem
import shutil
from get_task import get_task

hfs = HfFileSystem()

def run_classification(task_name, image_path, model_name=None, imgsize=384):
    print(f"Starting classification task '{task_name}' on image '{image_path}'")
    
    task = get_task(task_name)
    repository = task.repository
    
    # THIS LINE FOR DOWNLOADED ALL THE MODEL FROM EVERY EACH REPO
    # download_all_models(repository)
    
    model_name = model_name or task.default_model
    
    print(f"Loading image '{image_path}'")
    image = Image.open(image_path)
    
    print(f"Running classification task '{task_name}' with model '{model_name}'")
    result = task._gr_classification(image, model_name, imgsize)
    
    print("Classification complete. Printing results:")
    print_results_in_percentage(result)



def classify_images_in_folder(task_name, image_folder, output_folder, model_name=None, imgsize=384):
    task = get_task(task_name)
    model_name = model_name or task.default_model
    
    os.makedirs(output_folder, exist_ok=True)
    
    log_file = os.path.join(output_folder, "classification_log.txt")
    
    with open(log_file, "w") as log:
        for filename in os.listdir(image_folder):
            if filename.lower().endswith(('png', 'jpg', 'jpeg')):
                image_path = os.path.join(image_folder, filename)
                message = f"Processing image: {filename}\n"
                print(message)
                log.write(message)
                
                try:
                    image = Image.open(image_path)
                    image.verify()  # Verify that image is not corrupted
                    image = Image.open(image_path) 
                    
                    result = task._gr_classification(image, model_name, imgsize)
                    
                    highest_label = max(result, key=result.get)
                    highest_percentage = result[highest_label]
                    message = f"Image '{filename}' classified as '{highest_label}' with {highest_percentage:.2f}% confidence.\n"
                    print(message)
                    log.write(message)
                    
                    label_folder = os.path.join(output_folder, highest_label)
                    os.makedirs(label_folder, exist_ok=True)

                    destination_path = os.path.join(label_folder, filename)
                    shutil.move(image_path, destination_path)
                    message = f"Image '{filename}' moved to '{label_folder}'.\n"
                    print(message)
                    log.write(message)
                
                except (IOError, SyntaxError) as e:
                    # Handle corrupted images
                    message = f"Skipping corrupted image: {filename}. Error: {str(e)}\n"
                    print(message)
                    log.write(message)

