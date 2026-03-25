import tensorflow as tf  #type: ignore
from tensorflow.keras import layers, models  #type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator #type: ignore
import numpy as np  # type: ignore
import os

#Datenvorbereitung für das Train- und Validation Dataset
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2) #20% der Trainingsdaten werden für die Validierung verwendet

train_gen = train_datagen.flow_from_directory(
    r'C:\Users\Loran\Downloads\Dataset\Train',      
    target_size=(400, 270),
    batch_size=32, # Update: Batch size verdoppelt für beschleunigtes Training und Testen
    class_mode='binary', # Nur 2 klassen vorhanden also wird binary verwendet
    subset='training'
)

validation_gen = train_datagen.flow_from_directory(
    r'C:\Users\Loran\Downloads\Dataset\Train',
    target_size=(400, 270),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

#Datenvorbereitung für den Test (Test dataset getrennt vom Train dataset erstellen)
test_datagen = ImageDataGenerator(rescale=1./255)

test_gen = test_datagen.flow_from_directory(
    r'C:\Users\Loran\Downloads\Dataset\Test',
    target_size=(400, 270), #Bildgröße wird von 800x540 auf 400x270 verkleinert um Rechenleistung zu sparen aber genug Details zu behalten)
    batch_size=32,
    class_mode='binary', #subset nicht notwendig da es nur zum Testen verwendet wird anders als beim gesplitteten Train dataset
    shuffle=False
)


#Datenaugmentation definieren (für mehr Trainingsdaten und somit mehr Robustheit des Modells)
data_augmentation = tf.keras.Sequential([
    layers.RandomRotation(0.2),     #Drehung um 20%
    layers.RandomZoom(0.2),         #Zufälliger Zoom um 20%
    layers.RandomContrast(0.2),     #Kontrastanpassung um 20%
    #Update: Normalisierung entfernt, da die Reskalierung bereits durchgeführt wird
])


#Den eigentlichen CNN( Convolutional Neural Network) als Funktion definieren
def convolutional_layers():

    model = models.Sequential() #Update: model wieder vor Datenaugmentation gesetzt weil es sonst zu Fehlern kommt (vorher fiel mir das nicht auf)

    #Datenaugmentation wird dem Modell hinzugefügt 
    model.add(data_augmentation)      

    #Erste Convolutional Layer                                                              
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(400, 270, 3)))  #Muster suchen mit 32 Filtern der Größe 3x3
    model.add(layers.MaxPooling2D((2, 2))) # Reduziert die Größe der Ausgabe (Downsampling) um Rechenleistung zu sparen

    model.add(layers.Conv2D(64, (3, 3), activation='relu')) #Filteranzahl wird verdoppelt um komplexere Muster zu erkennen
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3), activation='relu')) #Filteranzahl wird verdoppelt um komplexere Muster zu erkennen
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(256, (3, 3), activation='relu')) #ReLu = Bildwerte mit einem Wert im Negativen werden auf 0 gesetzt und positive bleiben unverändert
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.GlobalAveragePooling2D())

    model.add(layers.Dense(128, activation='relu'))

    model.add(layers.Dense(1, activation='sigmoid')) # Z-Werte werden in Wahrscheinlichkeiten umgewandelt (0-1)
    
    model.compile(optimizer='adam',              # Adam optimizer wird oft verwendet und funktioniert gut           
                  loss='binary_crossentropy',    # Binary Crossentropy vergleicht vorhergesagte Werte mit tatsächlichen Werten im Binärsystem
                  metrics=['accuracy']
                  )
    
    return model 

#Ausgleichen der Klassenungleichheit im Trainingsdataset
class_weights = {0: 1.0, 1: 1.2} #Klasse 1(tumor) wird höher gewichtet, weil 20% in die Validation geht, jedoch gilt das auch für das test dataset was eigentlich ausgeglichen ist

#Early stopping um Overfitting (Überlernen) zu vermeiden
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=20, # wenn sich der CNN durch Validierungsverlust für 50 Epochen nicht verbessert wird das Training gestoppt
    restore_best_weights=True
)

#Checkpoint um das beste Modell zu speichern 
model_checkpoint = tf.keras.callbacks.ModelCheckpoint( 
 'SonoAI.keras', 
 monitor='val_loss', 
 save_best_only=True
 ) #Ein Dateipfad wird nicht benötigt da es da hin gespeichert wird wo das Skript ausgeführt wird

model = convolutional_layers()
model.fit(train_gen, epochs = 500, validation_data = validation_gen, callbacks=[early_stopping, model_checkpoint], class_weight=class_weights) 
    #Maximal 1000 Epochen aber Early_Stopping wird vorher stoppen und Model_checkpoint wird dieses Speichern 

#Testen des modells nach dem training
predictions = model.predict(test_gen) #Vorhersagen für das Test dataset generieren lassen
predictions = predictions.flatten() #Vorhersagen in eine flache Liste umwandeln
predicted_classes = [1 if probability > 0.5 else 0 for probability in predictions] #Wahrscheinlichkeiten/ in boolean Werten angeben lassen 
true_classes = test_gen.classes

# (Optional) Informationen über das Testen ausgeben 
error_count = sum(p != t for p, t in zip(predicted_classes, true_classes))
print(f'Anzahl der fehlerhaften Vorhersagen: {error_count} von {len(true_classes)}')

#Genauigkeit berechnen und ausgeben
accuracy = sum(p == t for p, t in zip(predicted_classes, true_classes)) / len(true_classes)
print(f'Testaccuracy: {accuracy * 100:.2f}%') 

#Confusion Matrix erstellen und ausgeben 
confusion_matrix = tf.math.confusion_matrix(true_classes, predicted_classes)
print('Confusion Matrix:' + str(confusion_matrix.numpy()))
