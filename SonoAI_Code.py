import tensorflow as tf # type: ignore
from tensorflow.keras import layers, models # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
import numpy as np # type: ignore
import os

#Datenvorbereitung 0 (train und validation datasets erstellen getrennt von test dataset weil Tensorflow das so will man keine ahnung muss einfach so sein (never change a running system brudi)) 
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_gen = train_datagen.flow_from_directory(
    'root_ordner/',       
    target_size=(256, 160),
    batch_size=32,
    class_mode='binary', # Nur 2 klassen vorhanden also wird binary verwendet
    subset='training'
)

validation_gen = train_datagen.flow_from_directory(
    'root_ordner/',
    target_size=(256, 160),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

#Datenvorbereitung 1(test dataset erstellen)
test_datagen = ImageDataGenerator(rescale=1./255)

test_gen = test_datagen.flow_from_directory(
    'root_ordner/',
    target_size=(256, 160),
    batch_size=32,
    class_mode='binary',
    shuffle=False
)


#Data Augmentation definieren (für mehr Trainingsdaten und somit mehr Robustheit des Modells)
#Bilder werden zufällig (Pseudorandom numbers) horizontal und vertikal gespiegelt und rotiert
data_augmentation1 = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
])

#Bilder werden zufällig (Pseudorandom numbers) vergrößert/verkleinert und der Kontrast wird angepasst
data_augmentation2 = tf.keras.Sequential([
    layers.RandomZoom(0.2),     
    layers.RandomContrast(0.2),
])


#Den eigentlichen CNN( Convolutional Neural Network) Programmieren
def convolutional_layers():

    #Datenaugmentation wird dem Modell hinzugefügt 
    model.add(data_augmentation1)       # Update: Augmentationsblock wurde nach oben verschoben statt wie vorher als zweites in der Funktion
    model.add(data_augmentation2)

    model = models.Sequential()

    #Erste Convolutional Layer
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 160, 3)))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))  #relu = rectified linear unit
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.GlobalAveragePooling2D())

    model.add(layers.Dense(128, activation='relu'))

    model.add(layers.Dense(1, activation='sigmoid')) # Sigmoid da nur 2 Klassen (Binary Classification)
    
    model.compile(optimizer='adam',              # Adam optimizer wird oft verwendet und funktioniert gut           
                  loss='binary_crossentropy',    # Binary Crossentropy da nur 2 Klassen 
                  metrics=['accuracy']
                  )
    
    return model


#Early stopping um Overfitting (Überlernen) zu vermeiden
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=10, 
    restore_best_weights=True
)

#Checkpoint um das beste Modell zu speichern
model_checkpoint = tf.keras.callbacks.ModelCheckpoint( 
'SonoAI.h5',
 monitor='val_loss', 
 save_best_only=True 
)

model = convolutional_layers()
model.fit(train_gen, epochs = 1000 , validation_data = validation_gen, callbacks=[early_stopping, model_checkpoint]) 
#Maximal 1000 Epochen aber Early_Stopping wird vorher stoppen und Model_checkpoint wird dieses Speichern 

#Testen des modells nach dem training
predictions = model.predict(test_gen)
predictions = predictions.flatten()
predicted_classes = [1 if probability > 0.5 else 0 for probability in predictions]
true_classes = test_gen.classes

# (Optional) Fehleranzahl ausgeben
error_count = sum(p != t for p, t in zip(predicted_classes, true_classes))
print(f'Anzahl der fehlerhaften Vorhersagen: {error_count} von {len(true_classes)}')

#Accuracy berechnen und ausgeben //update: Accuracy und Confusion Matrix hinzugefügt
accuracy = sum(p == t for p, t in zip(predicted_classes, true_classes)) / len(true_classes)
print(f'Testaccuracy: {accuracy * 100:.2f}%')

#Confusion Matrix erstellen und ausgeben
confusion_matrix = tf.math.confusion_matrix(true_classes, predicted_classes)
print('Confusion Matrix:' + str(confusion_matrix.numpy()))