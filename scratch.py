from psbody.mesh import Mesh
import os

def plytoobj():
    allfiles = os.listdir('../Experiments/')
    # pth = "../Experiments/20200811-151855/predictedPlys_checkpoint_414.pt"
    for folder in allfiles:
        pth = '../Experiments/'+folder+'/sample_plys'
        
        if os.path.exists(pth):
            folderlist = os.listdir(pth)
            cur = os.getcwd()
            os.chdir(pth)
            for file in folderlist:
                if file.startswith('.'):
                    os.remove(file)
                else : 
                    if file.endswith('.ply'):
                        mesh = Mesh(filename=file)
                        newfile = file.replace(".ply", ".obj")
                        mesh.write_obj(newfile)
                        print(f'{file} is replaced with {newfile}')
                        os.remove(file)

            os.chdir(cur)

# import shutil

# original = '/work/aifa/MeshAutoencoder/MySource/pytorch_coma/eval_config_parser.py'
# target = f'/work/aifa/MeshAutoencoder/MySource/Experiments/eval_config_parser.py'

# shutil.copyfile(original, target)

