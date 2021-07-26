from tkinter import filedialog
from tkinter import *

def openfile(directory: str,
             title: str = "Open CZI Image File",
             ftypename: str = "CZI Files",
             extension: str="*.czi") -> str:

    # request input and output image path from user
    root = Tk()
    root.withdraw()
    input_path = filedialog.askopenfile(title=title,
                                        initialdir=directory,
                                        filetypes = [(ftypename, extension)])
    if input_path is not None:
        return input_path.name
    if input_path is None:
        return None