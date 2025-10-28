import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import pandas as pd

base_dir = "./data" 

target_size = (128, 128)

def verify_and_resize_image(path, target_size):
    img = Image.open(path).convert('RGB')
    if img.size != target_size:
        img = img.resize(target_size)
    return img

datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_generator = datagen.flow_from_directory(
    base_dir,
    target_size=target_size,
    batch_size=8,
    class_mode='categorical',
    subset='training',
    shuffle=True
)
test_generator = datagen.flow_from_directory(
    base_dir,
    target_size=target_size,
    batch_size=8,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

base_model = MobileNetV2(input_shape=(128, 128, 3), include_top=False, weights='imagenet')
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(train_generator.num_classes, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=Adam(learning_rate=5e-4), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)

history1 = model.fit(train_generator, validation_data=test_generator, epochs=30, callbacks=[early_stop, lr_reduce])
history2 = []

base_model.trainable = True
for layer in base_model.layers[:100]:
    layer.trainable = False
model.compile(optimizer=Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

history2 = model.fit(train_generator, validation_data=test_generator, epochs=20, callbacks=[early_stop, lr_reduce])

loss, accuracy = model.evaluate(test_generator)
print(f"Acurácia final: {accuracy:.4f}, Perda final: {loss:.4f}")

os.makedirs('./docs', exist_ok=True)

true_labels = test_generator.classes
pred_probs = model.predict(test_generator)
pred_labels = np.argmax(pred_probs, axis=1)
cm = confusion_matrix(true_labels, pred_labels)
cmd = ConfusionMatrixDisplay(cm, display_labels=list(test_generator.class_indices.keys()))
cmd.plot(cmap=plt.cm.Blues)
plt.title("Matriz de Confusão")
fname = os.path.join('docs', 'confusion_matrix.png')
plt.savefig(fname, bbox_inches='tight')
plt.close()
print(f"Salvou matriz de confusão em: {fname}")

os.makedirs('./model', exist_ok=True)
model.save("./model/modelo_mobilenetv2_128x128_finetuned.keras")

def plot_history(histories, labels, out_dir='./docs'):
    os.makedirs(out_dir, exist_ok=True)
    for metric in ['accuracy', 'loss']:
        plt.figure()
        for hist, label in zip(histories, labels):
            plt.plot(hist.history[metric], label=f'{label} treino')
            plt.plot(hist.history['val_' + metric], label=f'{label} validação')
        plt.title(f'{metric.capitalize()} durante o treinamento')
        plt.xlabel('Épocas')
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.grid(True)
        fname = os.path.join(out_dir, f'{metric}_treinamento.png')
        plt.savefig(fname, bbox_inches='tight')
        plt.close()

if history2 :
    plot_history([history1, history2], ['Inicial', 'Fine-tuning'])
else:
    plot_history([history1], ['Inicial', 'Fine-tuning'])

os.makedirs('./docs/test_predictions', exist_ok=True)

Xi = 10
print(Xi, ' imgs')
filenames = test_generator.filepaths
sample_files = random.sample(filenames, Xi)

class_names = [None] * len(test_generator.class_indices)
for name, idx in test_generator.class_indices.items():
    class_names[idx] = name

previs = []

for i, file in enumerate(sample_files, start=1):
    img = verify_and_resize_image(file, target_size)
    img_array = img_to_array(img) / 255.0
    if img_array.ndim == 2:
        img_array = np.stack([img_array]*3, axis=-1)
    if img_array.shape[-1] == 1:
        img_array = np.concatenate([img_array]*3, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)
    try:
        pred = model.predict(img_array)
    except Exception as e:
        print(f"Erro ao prever arquivo {file}: {e}")
        continue

    pred_vec = pred[0]
    predicted_idx = int(np.argmax(pred_vec))
    predicted_name = class_names[predicted_idx]

    file_index = filenames.index(file)
    true_idx = int(test_generator.classes[file_index])
    true_name = class_names[true_idx]

    status = "ACERTO" if predicted_idx == true_idx else "ERRO"

    previs.append(pred_vec)

    plt.figure(figsize=(4,4))
    plt.imshow(img)
    plt.title(f"Previsão: {predicted_name} ({pred_vec[predicted_idx]:.2f})  |  Real: {true_name}  |  {status}", fontsize=9)
    plt.axis('off')

    safe_pred = predicted_name.replace(" ", "_")
    safe_true = true_name.replace(" ", "_")
    out_fname = os.path.join('docs/test_predictions', f'pred_{i:02d}_{status}_{safe_pred}_real_{safe_true}.png')
    plt.savefig(out_fname, bbox_inches='tight')
    plt.close()
    print(f"Salvou: {out_fname}")

if len(previs) == 0:
    print("Nenhuma previsão válida para gerar DataFrame.")
else:
    preds_arr = np.vstack(previs) 
    class_names = [None] * len(test_generator.class_indices)
    for name, idx in test_generator.class_indices.items():
        class_names[idx] = name
    df_prevs = pd.DataFrame(preds_arr, columns=class_names)
    print(df_prevs.head(10))