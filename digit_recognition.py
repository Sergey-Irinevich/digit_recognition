import tkinter as tk
from keras.models import load_model
from PIL import ImageGrab
from PIL import Image
import numpy as np


class Sketchpad(tk.Canvas):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.bind('<Button-1>', self.save_coord)
        self.bind('<B1-Motion>', self.draw_line)

    def save_coord(self, event):
        self.x_prev = event.x
        self.y_prev = event.y

    def draw_line(self, event):
        self.create_line(self.x_prev, self.y_prev, event.x, event.y, fill='black', width=20, capstyle='round',
                         smooth=True)
        self.save_coord(event)

    def clear_canvas(self):
        self.delete('all')


class Recognizer(object):
    def __init__(self):
        self.model = load_model('digital_model.h5')

    def recognize_digit(self):
        x1 = window.winfo_x() + sketch.winfo_x()
        y1 = window.winfo_y() + sketch.winfo_y()
        x2 = x1 + sketch.winfo_width()
        y2 = y1 + sketch.winfo_height()
        img = ImageGrab.grab(bbox=(x1, y1, x2, y2))
        img = img.resize(size=(28, 28), resample=Image.LANCZOS)
        im = np.array(img)
        im = np.bitwise_not(im)
        im = im.astype('float32')/255
        im = im[np.newaxis, :, :, 0, np.newaxis]
        classes = self.model.predict(im)
        answer = classes.argmax()
        label_answer['text'] = f'Это цифра {answer}'


window = tk.Tk()
window.title('Распознавание рукописных цифр')

sketch = Sketchpad(window, width='500', height='500', bg='white')
sketch.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky='nsew')

button_clear = tk.Button(text='Очистить', font='Times 14', command=sketch.clear_canvas)
button_clear.grid(row=1, column=0, padx=20, pady=20)

rec = Recognizer()
button_recognize = tk.Button(text='Распознать', font='Times 14', command=rec.recognize_digit)
button_recognize.grid(row=1, column=1, padx=20, pady=20)

label_answer = tk.Label(text='-', font='Times 16')
label_answer.grid(row=2, column=0, columnspan=2, padx=10, pady=10, sticky='nsew')

window.mainloop()
