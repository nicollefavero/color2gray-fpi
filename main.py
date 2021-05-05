import color2gray as c2g
from tkinter import * 
from PIL import Image, ImageTk, ImageDraw, ImageOps, ImageEnhance
import matplotlib.pyplot as plt 


class ImageSelectionView(Tk):
    def __init__(self):
        Tk.__init__(self)
        self.setupInterface()

    def setupInterface(self):

        # Dropdown menu options
        imageOptions = [
            "Sunset",
            "Butterfly",
            "Car",
            "Construction",
            "Hidden Number",
            "Island Map",
            "Square"
        ]

        # Setups dropdown menu
        rowFrame = Frame(self, bd=3)
        self.menuValue = StringVar(self)
        self.menuValue.set(imageOptions[0])
        self.imageOptionsDropDown = OptionMenu(rowFrame, self.menuValue, *imageOptions, command=self.changeImage)
        self.imageOptionsDropDown.pack(side='left')
        rowFrame.pack(fill='x')

        # Setups canvas
        self.canvas = Canvas(self, bd = 0, highlightthickness = 0, width = 400, height = 250)
        self.canvas.pack(fill = "both", expand = 1)
        self.originalImage = Image.open('images/sunset.png').convert('RGB')
        self.originalImage.thumbnail((512, 512))

        # Setups images
        self.photoshopGrayImage = Image.open('images/sunsetGray.png').convert('RGB')
        self.photoshopGrayImage.thumbnail((512, 512))

        c2g.colorToGray('images/sunset.png')
        self.colorToGrayImage = Image.open('output.png').convert('RGB')
        self.colorToGrayImage.thumbnail((512, 512))

        self.originalImageTk = ImageTk.PhotoImage(self.originalImage)
        self.photoshopGrayImageTk = ImageTk.PhotoImage(self.photoshopGrayImage)
        self.colorToGrayImageTk = ImageTk.PhotoImage(self.colorToGrayImage)

        self.originalImageItem = self.canvas.create_image(10, 30, anchor = 'nw', image = self.originalImageTk)
        self.photoshopGrayImageItem = self.canvas.create_image(self.originalImage.size[0] + 20, 30, anchor = 'nw', image = self.photoshopGrayImageTk)
        self.colorToGrayImageItem = self.canvas.create_image(self.originalImage.size[0] + self.photoshopGrayImage.size[0] + 40, 30, anchor = 'nw', image = self.colorToGrayImageTk)

        self.canvas.config(width = self.originalImage.size[0] * 3 + 50, height = self.originalImage.size[1] * 1.3)

    def changeImage(self, selectedOption):
        imagePath = ''
        if selectedOption == 'Sunset':
            imagePath = 'images/sunset'
        elif selectedOption == 'Butterfly':
            imagePath = 'images/butterfly'
        elif selectedOption == 'Car':
            imagePath = 'images/car'
        elif selectedOption == 'Construction':
            imagePath = 'images/construction'
        elif selectedOption == 'Hidden Number':
            imagePath = 'images/hiddenNumber'
        elif selectedOption == 'Island Map':
            imagePath = 'images/mapIsland'
        elif selectedOption == 'Square':
            imagePath = 'images/square'
        
        extension = '.png'

        # Updates the displayed images
        c2g.colorToGray(imagePath + extension)
        self.newColorToGrayImage = Image.open('output.png').convert('RGB')
        self.newColorToGrayTkImage = ImageTk.PhotoImage(self.newColorToGrayImage)
        self.canvas.itemconfig(self.colorToGrayImageItem, image=self.newColorToGrayTkImage)

        self.newOriginalImage = Image.open(imagePath + extension).convert('RGB')
        self.newOriginalTkImage = ImageTk.PhotoImage(self.newOriginalImage)
        self.canvas.itemconfig(self.originalImageItem, image=self.newOriginalTkImage)

        self.newPhotoshopImage = Image.open(imagePath + 'Gray' + extension).convert('RGB')
        self.newPhotoshopTkImage = ImageTk.PhotoImage(self.newPhotoshopImage)
        self.canvas.itemconfig(self.photoshopGrayImageItem, image=self.newPhotoshopTkImage)


imageSelectionView = ImageSelectionView()
imageSelectionView.mainloop()