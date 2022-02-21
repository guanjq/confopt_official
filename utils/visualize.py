import py3Dmol
from rdkit import Chem


def MolTo3DView(mol, size=(300, 300), style="stick", surface=False, opacity=0.5):
    """Draw molecule in 3D

    Args:
    ----
        mol: rdMol, molecule to show
        size: tuple(int, int), canvas size
        style: str, type of drawing molecule
               style can be 'line', 'stick', 'sphere', 'carton'
        surface, bool, display SAS
        opacity, float, opacity of surface, range 0.0-1.0
    Return:
    ----
        viewer: py3Dmol.view, a class for constructing embedded 3Dmol.js views in ipython notebooks.
    """
    assert style in ('line', 'stick', 'sphere', 'carton')

    viewer = py3Dmol.view(width=size[0], height=size[1])
    if isinstance(mol, list):
        for i, m in enumerate(mol):
            mblock = Chem.MolToMolBlock(m)
            viewer.addModel(mblock, 'mol' + str(i))
    elif len(mol.GetConformers()) > 1:
        for i in range(len(mol.GetConformers())):
            mblock = Chem.MolToMolBlock(mol, confId=i)
            viewer.addModel(mblock, 'mol' + str(i))
    else:
        mblock = Chem.MolToMolBlock(mol)
        viewer.addModel(mblock, 'mol')
    viewer.setStyle({style: {}})
    if surface:
        viewer.addSurface(py3Dmol.SAS, {'opacity': opacity})
    viewer.zoomTo()
    return viewer
