import sys
from PyQt6.QtCore import Qt,pyqtSignal,pyqtSlot
from PyQt6.QtGui import *
from PyQt6.QtWidgets import *
from argparse import ArgumentParser
from pathlib import Path
from dataclasses import dataclass
from typing import List
from wff_common import db_get_faces
from PIL import ImageQt

def get_args():
    parser = ArgumentParser()
    parser.add_argument( '--imgroot',required=True,help='Root of added images')
    return parser.parse_args()

@dataclass
class WFFImage:
    path:Path

class WFFImageGridLabel(QLabel):
    gridImageClicked = pyqtSignal(int,int,str)
    def __init__(self,image:Path,row,col):
        QLabel.__init__(self,"grid item")
        self.setToolTip(f"Source: {image.name}")
        self.file = image
        self.row = row
        self.col = col
    def mousePressEvent(self,event):
        self.gridImageClicked.emit(self.row,self.col,str(self.file))

class WFFFaceGroupWindow(QWidget):
    pass

class WFFPictureMatchesWindow(QWidget):
    def __init__(self, image:str): 
        QWidget.__init__(self)
        layout = QGridLayout()
        print(f"Picture Matches: {image}")
        px = QPixmap(str(image)) 
        spx=px.scaled(250,250,Qt.AspectRatioMode.KeepAspectRatio)
        label=QLabel("image")
        label.setPixmap(spx)

        # returns array of PIL.Image
        faces = db_get_faces(Path(image).name)
        vbox = QVBoxLayout()

        vbox.addWidget(QLabel("test1"))
        vbox.addWidget(QLabel("test2"))
        vbox.addWidget(QLabel("test3"))
        layout.addWidget(label,0,0,3,2)
        layout.addItem(vbox,0,2,3,1)
        self.setLayout(layout)

class WFFPictureGridWindow(QWidget):
    imageSelected  = pyqtSignal(str)

    def __init__(self, images:List[WFFImage]):
        QWidget.__init__(self)
        layout = QGridLayout()
        print(f"Displaying {len(images)}")
        for r in range(0,len(images),3):
            # in the row, it will be r + c = pos
            for c in range(min(len(images)-r,3)):
                print(f"idx {c+r}")
                row=int(r/3)
                img = images[c+r]
                print(f"Row {row}, Column {c} : {img.path.name}")
                px = QPixmap(str(img.path))
                spx=px.scaled(100,100,Qt.AspectRatioMode.KeepAspectRatio)
                px_label = WFFImageGridLabel(img.path,row,c)
                px_label.gridImageClicked.connect(self.imagePicked )
                #self.imagePicked.connect(px_label.gridImageClicked)
                px_label.setPixmap(spx)
                #px_label.scaledContents = True
                layout.addWidget(px_label,row,c)
        self.setLayout(layout)

    @pyqtSlot(int,int,str)
    def imagePicked(self,row,col,image_name):
        print(f"{row} {col}: Load {image_name}")
        self.imageSelected.emit(image_name)

class WFFWindow(QWidget):
    def __init__(self, images:List[WFFImage]):
        QWidget.__init__(self)
        self.setWindowTitle("Wassmann Family Finder")
        self.setGeometry(500,500,500,500)
        self.picture_grid = WFFPictureGridWindow(images)
        self.picture_grid.imageSelected.connect(self.viewImage)
        layout = QVBoxLayout()
        layout.addWidget(self.picture_grid)
        self.setLayout(layout)

        self.image_views = {}

    @pyqtSlot(str)
    def viewImage(self,image_name):
        print(f"main: Load {image_name}")
        if image_name not in self.image_views:
            self.image_views[image_name] = WFFPictureMatchesWindow(image_name)
        self.picture_grid.hide()
        self.layout().replaceWidget(self.picture_grid,self.image_views[image_name])


class WFFImageGried(QWidget):
    def __init__(self,images):
        QWidget.__init__()

if __name__ == "__main__":
    args = get_args()
    base = Path(args.imgroot)
    if not base.is_dir():
        print(f"{base} isn't a directory")
        sys.exit(1)

    images=[]
    for f in base.glob("*.jpg"):
        print(f"* {f}")
        images.append(WFFImage(path=f))

    if len(images) == 0:
        print(f"{base} had no .jpg images") 
        sys.exit(1)

    app = QApplication(sys.argv)
    win = WFFWindow( images)
    win.show()

    app.exec()
