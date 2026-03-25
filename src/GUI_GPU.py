import os
import tkinter as tk
from tkinter import filedialog
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

#Pfad zum gespeicherten Modell
model_path = r'C:\Users\Loran\Documents\JuFo_Sono1\SonoAI_CPU.h5'

#Fenster Erstellen mit Tkinter
Window = tk.Tk()

#Fenstertitel festlegen
Window.title("SonoAI-CPU Version")

#Fensterdetails festlegen
Canvas = tk.Canvas(Window, width=500, height=500, bg="#000011")
Canvas.pack(expand=True, fill='both')

#Titeltext hinzufügen
Text = tk.Label(Window, text="SonoAI - Erkennung von Fetalen Hirntumoren in Ultraschallbildern", bg="#000011", fg="white", font=("Comforta", 12))
Text.place(relx=0.5, rely=0.05, anchor='center')

Text2 = tk.Label(Window, text="Bitte wählen sie eine Bilddatei aus", bg="#000011", fg="white", font=("Comforta", 10))
Text2.place(relx=0.5, rely=0.5, anchor='center')

#Icon festlegen
Window.iconbitmap(r'C:\Users\Loran\Pictures\Saved Pictures\SonoAI_Icon.ico')


#Dateiauswahl-Dialog öffnen
file_path = filedialog.askopenfilename(title="Ultraschallbild auswählen", filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*")])
    
if file_path:
        #Modell laden
        model = load_model(model_path)
        
        #Bild vorbereiten
        img = image.load_img(file_path, target_size=(400, 270))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Batch-Dimension hinzufügen
        img_array /= 255.0  # Reskalierung

        #Vorhersage machen
        prediction = model.predict(img_array)
        
        #Ergebnis interpretieren
        if prediction[0][0] > 0.5:
            result_text = "Tumor erkannt"
        else:
            result_text = "Kein Tumor erkannt"
        
        #Ergebnis anzeigen
        Canvas.delete("all")  # Vorherige Inhalte löschen
        Result_Label = tk.Label(Window, text=result_text, bg="#000011", fg="white", font=("Comforta", 16))
        Result_Label.place(relx=0.5, rely=0.7, anchor='center')


Window.mainloop()