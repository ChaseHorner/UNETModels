import os


''' Saves model output results from local model folder to shared resfs folder.'''

output_folder = '/resfs/GROUPS/KBS/kars_yield/model_outputs'

def save_resfs(model_folder, model_name):
        
    for filename in os.listdir(model_folder):
        if filename.endswith('.pt') or filename.endswith('.out') or filename.endswith('.err'):
            continue
        src = os.path.join(model_folder, filename)
        dst = os.path.join(output_folder, model_name, filename)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.isfile(src):
            with open(src, 'rb') as fsrc, open(dst, 'wb') as fdst:
                fdst.write(fsrc.read())