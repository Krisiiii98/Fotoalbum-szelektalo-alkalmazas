import cv2
import os.path
import tkinter as tk
import customtkinter
import ctypes
from tkinter import *
from tkinter import ttk
from tkinter import filedialog
from PIL import ImageTk, Image 
from ultralytics import YOLO

from faults_detection import *
from folderselect import ImageStore
from classification import *
from transformations import *

gui = tk.Tk()

class canvas_data:
    canvas_sizex = 830
    canvas_sizey = 560

class image_typ:
    purple_img = []
    hist_img = []
    repaired_img = []
    original_img = []

class image_position:
    position = 0

class image_faults:
    noise = False
    exposed = False
    blurry = False
    redeye = False
    purple = False
    fault_counter = 0

class image_features:
    quality = 0
    score = 0
    eyeRects = 0
    blur_type = ""

class checkbox:
    under_over_exposure = tk.IntVar()
    blur = tk.IntVar()
    red_eye = tk.IntVar()
    purple_fringe = tk.IntVar()

def image_resizing(image, x, y):
    img_x = image.shape[1]
    img_y = image.shape[0]
    if img_x > img_y and img_x > x:
        f = x / img_x
        if (img_y * f) > y:
            f = y / img_y
            return f
        return f
    elif img_y > img_x and img_y > y:
        f = y / img_y
        if (img_x * f) > x:
            f = x / img_x
            return f
        return f
    else:
        f = 1
        return f

#képek betöltése a mappából
def image_loading(images, canvas):
    textbox1.delete("1.0","end")
    if images is not None:
        
        position = image_position.position

        if position <= len(images):
            img = cv2.cvtColor(images[position], cv2.COLOR_BGR2RGB)
            image_typ.original_img = img
            img = cv2.resize(img, None, fx=image_resizing(img, canvas_data.canvas_sizex, canvas_data.canvas_sizey), fy=image_resizing(img, canvas_data.canvas_sizex, canvas_data.canvas_sizey))
            filename =ImageTk.PhotoImage(Image.fromarray(img))
            canvas.image = filename
            canvas.create_image(0,0,anchor='nw',image=filename)
    next_image['state'] = tk.DISABLED
    og_save['state'] = tk.DISABLED
    repair['state'] = tk.DISABLED
    image_switch.deselect()
    image_switch.configure(state=DISABLED)
    switch_label.configure(text="Eredeti")
    new_save['state'] = tk.DISABLED
    detection['state'] = tk.NORMAL
    checkbox1['state'] = tk.DISABLED
    checkbox2['state'] = tk.DISABLED
    checkbox3['state'] = tk.DISABLED
    checkbox4['state'] = tk.DISABLED
    checkbox.under_over_exposure.set(0)
    checkbox.blur.set(0)
    checkbox.purple_fringe.set(0)
    checkbox.red_eye.set(0)

def fault_detection(images):
    detection['state'] = tk.DISABLED   
    img = images[image_position.position]
    image_faults.fault_counter = 0

    image_faults.noise = detect_noise_fft(img)
    if image_faults.noise is True:
        textbox1.insert(END, "A kép zajos!")
    else:
        textbox1.insert(END, "Nincs jelentős zaj, nem szükséges a zajszűrés.")

    image_faults.exposed, image_features.quality, _, _, _ = estimate_exposure(img)
    if image_faults.exposed is True and image_features.quality == 1:
        textbox1.insert(END, "\nalulexponált")
        checkbox1['state'] = tk.NORMAL
        image_faults.fault_counter += 1    
    elif image_faults.exposed is True and image_features.quality == 2: 
        textbox1.insert(END, "\ntúlexponált")
        checkbox1['state'] = tk.NORMAL
        image_faults.fault_counter += 1
    else:
        textbox1.insert(END, "\nJól exponált")
    progress_step(4)
    gui.update_idletasks()

    image_faults.blurry, image_features.score, image_features.blur_type = estimate_blur(img)
    if image_faults.blurry is True:
        textbox1.insert(END, "\nhomályos")
        checkbox2['state'] = tk.NORMAL
        image_faults.fault_counter += 1
        if image_features.score < 30:
            textbox1.insert(END, "\nA kép nem javítható! Lépjen a következő képre.")
            next_image['state'] = tk.NORMAL
            image_position.position += 1
            end_of_images(ImageStore.images, image_position.position)
            return
    else:
        textbox1.insert(END, "\nnem homályos")
    progress_step(4)
    gui.update_idletasks()

    detected_faces = face_detection(img)
    if detected_faces > 0:
        image_faults.redeye, image_features.eyeRects = estimate_redeye(img)
        if image_faults.redeye is True:
            textbox1.insert(END, "\nPiros szem hiba")
            checkbox3['state'] = tk.NORMAL
            image_faults.fault_counter += 1
            progress_step(4)
            gui.update_idletasks()
        else:
            textbox1.insert(END, "\nNincs piros szem")
        progress_step(4)
        gui.update_idletasks()
    else:
        textbox1.insert(END, "\nNincs piros szem")
    progress_step(4)
    gui.update_idletasks()

    image_faults.purple, image_typ.purple_img, _  = purple_fringe_detection(img)
    if image_faults.purple is True:
        textbox1.insert(END, "\nPurple fringe")
        checkbox4['state'] = tk.NORMAL
        image_faults.fault_counter += 1
    else:
        textbox1.insert(END, "\nNincs purple fringe")

    if image_faults.fault_counter > 0:
        textbox1.insert(END, "\nJavításra van szükség")
        repair['state'] = tk.NORMAL
    else:
        textbox1.insert(END, "\nNincs szükség javításra")
        og_save['state'] = tk.NORMAL
        next_image['state'] = tk.NORMAL
    progress_step(4)
    gui.update_idletasks()

def faultimage_correction(images):
    progress['value'] = 0
    gui.update_idletasks()
    repair['state'] = tk.DISABLED
    img = images[image_position.position]

    if image_faults.noise == True:
        img = cv2.bilateralFilter(img, 9, 75, 75)

    if checkbox.blur.get() == 1:
        image_faults.blurry, image_features.score, image_features.blur_type = estimate_blur(img)
        if image_faults.blurry == True:
            img = blur_correction(img, image_features.blur_type)
            progress_step(image_faults.fault_counter)
            gui.update_idletasks()
            lap_var = laplacian_variance(img)
            print(lap_var)

    if checkbox.under_over_exposure.get() == 1:
        image_faults.exposed, image_features.quality, _, _, _ = estimate_exposure(img)
        if image_faults.exposed == True:
            img = exposure_correction(img)
            progress_step(image_faults.fault_counter)
            gui.update_idletasks()

    if checkbox.red_eye.get() == 1:
        image_faults.redeye, image_features.eyeRects = estimate_redeye(img)
        if image_faults.redeye == True:
            img = Red_eye_correction(img, image_features.eyeRects)
            progress_step(image_faults.fault_counter)
            gui.update_idletasks()

    if checkbox.purple_fringe.get() == 1:
        image_faults.purple, image_typ.purple_img, _  = purple_fringe_detection(img)
        if image_faults.purple == True:
            img = purple_fringe_correction(img, image_typ.purple_img)
            progress_step(image_faults.fault_counter)
            gui.update_idletasks()

    og_save['state'] = tk.NORMAL
    new_save['state'] = tk.NORMAL
    image_switch.configure(state=NORMAL)
    
    image_typ.repaired_img = img

def newimg_savefile():
    progress['value'] = 0
    gui.update_idletasks()
    og_save['state'] = tk.DISABLED
    new_save['state'] = tk.DISABLED
    classify_path = classify(ImageStore.gallery_path, image_typ.repaired_img)
    cv2.imwrite(os.path.join(classify_path, ImageStore.image_names[image_position.position]), image_typ.repaired_img)
    next_image['state'] = tk.NORMAL
    image_position.position += 1
    end_of_images(ImageStore.images, image_position.position)


def originalimg_savefile():
    progress['value'] = 0
    gui.update_idletasks()
    og_save['state'] = tk.DISABLED
    new_save['state'] = tk.DISABLED
    #new_img_view['state'] = tk.DISABLED
    image_switch.configure(state=DISABLED)
    cv2.imwrite(os.path.join(ImageStore.gallery_path, ImageStore.image_names[image_position.position]), image_typ.original_img)
    next_image['state'] = tk.NORMAL
    image_position.position += 1
    end_of_images(ImageStore.images, image_position.position)

def switcher():
    switch_label.configure(text=switch_var.get())

    if switch_label.cget("text") == "Szerkesztett":
        act_img = image_typ.repaired_img
        act_img = cv2.cvtColor(act_img, cv2.COLOR_BGR2RGB)
        act_img = cv2.resize(act_img, None, fx=image_resizing(act_img, canvas_data.canvas_sizex, canvas_data.canvas_sizey), fy=image_resizing(act_img, canvas_data.canvas_sizex, canvas_data.canvas_sizey))
        filename =ImageTk.PhotoImage(Image.fromarray(act_img))
        canvas.image = filename
        canvas.create_image(0,0,anchor='nw',image=filename)
    else:
        act_img = image_typ.original_img
        act_img = cv2.resize(act_img, None, fx=image_resizing(act_img, canvas_data.canvas_sizex, canvas_data.canvas_sizey), fy=image_resizing(act_img, canvas_data.canvas_sizex, canvas_data.canvas_sizey))
        filename =ImageTk.PhotoImage(Image.fromarray(act_img))
        canvas.image = filename
        canvas.create_image(0,0,anchor='nw',image=filename)

def progress_step(task_count):
    step_value = 99.9 / task_count
    progress['value'] += step_value
    gui.update_idletasks()

def end_of_images(images, position):
    if position == len(images):
        next_image.pack_forget()
        next_image.destroy()
        quit = tk.Button(gui,text='Bezárás', command=lambda: on_closing(), bg='#364156', foreground='white', font=('arial', 10, 'bold'))
        quit.place(x=1120,y=25)

def on_closing():
    exit(0)

gui.geometry('1280x680')
gui.title('Fotóalbum szelektáló')
gui.configure(background='#CDCDCD')
gui.resizable(True, True)

heading = Label(gui, text="Fotóalbum szelektáló", pady=20, font=('arial', 24, 'bold'))
heading.configure(background='#CDCDCD', foreground='#364156')
heading.pack(side=TOP)

label=Label(gui,background='#CDCDCD', font=('arial', 15, 'bold'))
label.pack(side=BOTTOM, expand=True)

canvas = Canvas(gui, width=830, height=560, background='#CDCDCD')
canvas.pack(side=LEFT)

next_image = tk.Button(gui,text='Következő kép', command=lambda: image_loading(ImageStore.images, canvas),bg='#364156', foreground='white', font=('arial', 10, 'bold'))
next_image.place(x=1120,y=25)
detection = tk.Button(gui,text='Detektálás', command=lambda: fault_detection(ImageStore.images),bg='#364156', foreground='white', font=('arial', 10, 'bold'))
detection.place(x=1180,y=150)
repair = tk.Button(gui,text='Javítás', command=lambda: faultimage_correction(ImageStore.images), bg='#364156', foreground='white', font=('arial', 10, 'bold'))
repair.place(x=875,y=450)
og_save = tk.Button(gui,text='Eredeti mentése', command=lambda: originalimg_savefile(),bg='#364156', foreground='white', font=('arial', 10, 'bold'))
og_save.place(x=875,y=610)
new_save = tk.Button(gui,text='Új mentése', command=lambda: newimg_savefile(),bg='#364156', foreground='white', font=('arial', 10, 'bold'))
new_save.place(x=1000,y=610)
switch_label = customtkinter.CTkLabel(gui, text = "Eredeti")
switch_label.place(x=925, y=530)
switch_var = customtkinter.StringVar(value="on")
image_switch = customtkinter.CTkSwitch(gui, text="", command=lambda: switcher(), variable = switch_var,  onvalue = "Szerkesztett", offvalue = "Eredeti", state=DISABLED, width = 50)
image_switch.place(x=875, y=530)

checkbox1 = tk.Checkbutton(gui, text='Alul-/Túlexponálás',variable=checkbox.under_over_exposure, onvalue=1, offvalue=0, background='#CDCDCD', state=DISABLED)
checkbox1.place(x=870,y=360)
checkbox2 = tk.Checkbutton(gui, text='Homályosság',variable=checkbox.blur, onvalue=1, offvalue=0, background='#CDCDCD', state=DISABLED)
checkbox2.place(x=870,y=380)
checkbox3 = tk.Checkbutton(gui, text='Piros szem',variable=checkbox.red_eye, onvalue=1, offvalue=0, background='#CDCDCD', state=DISABLED)
checkbox3.place(x=870,y=400)
checkbox4 = tk.Checkbutton(gui, text='Purple fringe',variable=checkbox.purple_fringe, onvalue=1, offvalue=0, background='#CDCDCD', state=DISABLED)
checkbox4.place(x=870,y=420)

textbox1 = tk.Text(gui, height=8, width=55, bg='white', font=('arial', 10), state="normal")
textbox1.place(x=870, y=180)

gui.wm_attributes('-transparentcolor', '#ab23ff')

Label(gui, text="Hibák detektálása", background='#CDCDCD', foreground='#364156', font=('arial', 16, 'bold')).place(x=870,y=150)
Label(gui, text="Javítás", background='#CDCDCD', foreground='#364156', font=('arial', 16, 'bold')).place(x=870,y=330)
Label(gui, text="Szerkesztett kép megtekintése", background='#CDCDCD', foreground='#364156', font=('arial', 16, 'bold')).place(x=870,y=500)
Label(gui, text="Kép mentése", background='#CDCDCD', foreground='#364156', font=('arial', 16, 'bold')).place(x=870,y=580)
prog_label = Label(gui, text="", background='#ab23ff', foreground='#364156', font=('arial', 12, 'bold'))
prog_label.place(x=1180, y=615)

progress = ttk.Progressbar(gui, orient = 'horizontal', length = 100)
progress.place(x=1160, y=615)

gui.protocol("WM_DELETE_WINDOW", on_closing)

image_loading(ImageStore.images, canvas)
gui.mainloop()

cv2.waitKey(0)