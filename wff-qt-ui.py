import sys
from PyQt6.QtGui import *
from PyQt6.QtWidgets import *
from argparse import ArgumentParser
from pathlib import Path
from dataclasses import dataclass
from typing import List

def get_args():
    parser = ArgumentParser()
    parser.add_argument( '--imgroot',required=True,help='Root of added images')
    return parser.parse_args()

@dataclass
class WFFImage:
    path:Path

class WFFWindow(QWidget):
    def __init__(self, images:List[WFFImage]):
        QWidget.__init__(self)
        self.setWindowTitle("Wassmann Family Finder")
        self.setGeometry(500,500,500,500)
        layout = QGridLayout()
        print(f"Displaying {len(images)}")
        # r rows which can have at max 3 images.
        for r in range(0,len(images),3):
            # in the row, it will be r + c = pos
            for c in range(min(len(images)-r,3)):
                print(f"idx {c+r}")
                row=int(r/3)
                img = images[c+r]
                print(f"Row {row}, Column {c} : {img.path.name}")
                px = QPixmap(str(img.path))
                px_label = QLabel(img.path.name)
                px_label.setPixmap(px)
                layout.addWidget(px_label,row,c)
        self.setLayout(layout)


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
