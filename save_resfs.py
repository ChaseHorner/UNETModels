import os


output_folder = '/resfs/GROUPS/KBS/kars_yield/model_outputs'

def save_resfs(model_folder, model_name):
        
    for filename in os.listdir(model_folder):
        if filename.endswith('.pt'):
            continue
        src = os.path.join(model_folder, filename)
        dst = os.path.join(output_folder, model_name, filename)
        if os.path.isfile(src):
            with open(src, 'rb') as fsrc, open(dst, 'wb') as fdst:
                fdst.write(fsrc.read())