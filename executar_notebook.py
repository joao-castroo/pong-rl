"""Executa o próprio notebook e salva as saídas a cada célula. Não executa a sessão humana."""
from pathlib import Path
import os, json, traceback
root=Path(__file__).resolve().parent
os.chdir(root)
for key, folder in [('IPYTHONDIR','.ipython'),('JUPYTER_CONFIG_DIR','.jupyter'),('JUPYTER_DATA_DIR','.jupyter/data'),('JUPYTER_RUNTIME_DIR','.jupyter/runtime'),('MPLCONFIGDIR','.matplotlib')]:
    os.environ[key]=str(root/folder)
    (root/folder).mkdir(parents=True,exist_ok=True)
import nbformat
from nbclient import NotebookClient
nbpath=root/'Pong_RL.ipynb'
nb=nbformat.read(nbpath,as_version=4)
client=NotebookClient(nb,timeout=None,kernel_name='python3',resources={'metadata':{'path':str(root)}})
state=root/'artifacts/execution_status.json'
state.parent.mkdir(exist_ok=True)
try:
    with client.setup_kernel():
        for i,cell in enumerate(nb.cells):
            if cell.cell_type!='code' or 'tensorboard' in cell.metadata.get('tags',[]): continue
            state.write_text(json.dumps({'state':'running','cell':i,'source':cell.source.splitlines()[0]}))
            print(f'Executando célula {i}: {cell.source.splitlines()[0]}',flush=True)
            client.execute_cell(cell,i)
            nbformat.write(nb,nbpath)
    state.write_text(json.dumps({'state':'completed'}))
except BaseException as exc:
    nbformat.write(nb,nbpath)
    state.write_text(json.dumps({'state':'error','error':str(exc)}))
    traceback.print_exc()
    raise
