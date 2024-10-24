import os
import urllib.request
# Lista de clases para descargar
classes = ['cat', 'dog' , 'apple', 'tree', 'bird', 'airplane', 'key', 'bicycle', 'banana', 'fish', 'eye', 'car']
# Asegurarse de que la carpeta existe
if not os.path.exists('/data'):
    os.makedirs('/data')

def download():
  base = 'https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/'
  for c in classes:
    cls_url = c.replace('_', '%20')
    path = base+cls_url+'.npy'
    print(path)
    save_path = os.path.join('/app/project/data/REDES_NEURONALES/dereck/draws', c + '.npy')
    urllib.request.urlretrieve(path, save_path)
    print(f'Archivo guardado en: {save_path}')


download()
