import os
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from datetime import datetime

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

class_names = [None] * len(test_generator.class_indices)
for name, idx in test_generator.class_indices.items():
    class_names[idx] = name

cm = confusion_matrix(true_labels, pred_labels)
class_report = classification_report(true_labels, pred_labels, target_names=class_names, digits=4)

from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
precision, recall, f1, support = precision_recall_fscore_support(true_labels, pred_labels, average=None)
precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(true_labels, pred_labels, average='macro')
precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(true_labels, pred_labels, average='weighted')

from sklearn.preprocessing import label_binarize
true_labels_bin = label_binarize(true_labels, classes=range(len(class_names)))
auc_scores = []
for i in range(len(class_names)):
    try:
        auc = roc_auc_score(true_labels_bin[:, i], pred_probs[:, i])
        auc_scores.append(auc)
    except:
        auc_scores.append(0.0)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
report_path = os.path.join('docs', f'training_report_{timestamp}.txt')

with open(report_path, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("RELATÓRIO COMPLETO DE TREINAMENTO - REDE NEURAL PARA CLASSIFICAÇÃO DE TUMORES\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Data e Hora: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}\n\n")
    
    f.write("=" * 80 + "\n")
    f.write("1. ARQUITETURA DO MODELO\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Modelo Base: MobileNetV2 (pré-treinado com ImageNet)\n")
    f.write(f"Dimensões de Entrada: {target_size[0]}x{target_size[1]}x3 (RGB)\n")
    f.write(f"Número de Classes: {len(class_names)}\n")
    f.write(f"Classes: {', '.join(class_names)}\n\n")
    
    f.write("Camadas Adicionais:\n")
    f.write("  - GlobalAveragePooling2D\n")
    f.write("  - Dense(128, activation='relu')\n")
    f.write("  - Dense({}, activation='softmax') - Camada de saída\n\n".format(len(class_names)))
    
    f.write(f"Total de Parâmetros: {model.count_params():,}\n")
    trainable_count = sum([np.prod(v.shape.as_list()) for v in model.trainable_weights])
    non_trainable_count = sum([np.prod(v.shape.as_list()) for v in model.non_trainable_weights])
    f.write(f"Parâmetros Treináveis: {trainable_count:,}\n")
    f.write(f"Parâmetros Não-Treináveis: {non_trainable_count:,}\n\n")
    
    f.write("Detalhamento das Camadas:\n")
    f.write("-" * 80 + "\n")
    for i, layer in enumerate(model.layers):
        f.write(f"Camada {i}: {layer.name}\n")
        f.write(f"  Tipo: {layer.__class__.__name__}\n")
        if hasattr(layer, 'output_shape'):
            f.write(f"  Shape de Saída: {layer.output_shape}\n")
        if hasattr(layer, 'trainable'):
            f.write(f"  Treinável: {layer.trainable}\n")
        param_count = layer.count_params()
        if param_count > 0:
            f.write(f"  Parâmetros: {param_count:,}\n")
        f.write("\n")
    
    f.write("\n" + "=" * 80 + "\n")
    f.write("2. CONFIGURAÇÃO DE TREINAMENTO\n")
    f.write("=" * 80 + "\n\n")
    f.write("Fase 1 - Treinamento Inicial (Base Model congelada):\n")
    f.write("  - Épocas: 30\n")
    f.write("  - Learning Rate: 5e-4\n")
    f.write("  - Otimizador: Adam\n")
    f.write("  - Batch Size: 8\n")
    f.write("  - Early Stopping: patience=10, monitor='val_loss'\n")
    f.write("  - ReduceLROnPlateau: factor=0.5, patience=3\n\n")
    
    f.write("Fase 2 - Fine-tuning (Primeiras 100 camadas congeladas):\n")
    f.write("  - Épocas: 20\n")
    f.write("  - Learning Rate: 1e-5\n")
    f.write("  - Otimizador: Adam\n")
    f.write("  - Camadas treináveis a partir da camada 100\n\n")
    
    f.write("Data Augmentation:\n")
    f.write("  - Rescale: 1./255\n")
    f.write("  - Validation Split: 20%\n")
    f.write("  - Shuffle: True (training)\n\n")
    
    f.write("Informações do Dataset:\n")
    f.write(f"  - Total de amostras de treino: {train_generator.samples}\n")
    f.write(f"  - Total de amostras de validação: {test_generator.samples}\n")
    f.write(f"  - Distribuição por classe (validação):\n")
    for class_name, class_idx in test_generator.class_indices.items():
        count = np.sum(test_generator.classes == class_idx)
        percentage = (count / len(test_generator.classes)) * 100
        f.write(f"    * {class_name}: {count} amostras ({percentage:.2f}%)\n")
    
    f.write("\n" + "=" * 80 + "\n")
    f.write("3. MÉTRICAS DE DESEMPENHO (CONJUNTO DE VALIDAÇÃO)\n")
    f.write("=" * 80 + "\n\n")
    
    f.write("Métricas Globais:\n")
    f.write(f"  - Acurácia: {accuracy:.4f} ({accuracy*100:.2f}%)\n")
    f.write(f"  - Perda (Loss): {loss:.4f}\n")
    f.write(f"  - Precisão Macro: {precision_macro:.4f}\n")
    f.write(f"  - Recall Macro: {recall_macro:.4f}\n")
    f.write(f"  - F1-Score Macro: {f1_macro:.4f}\n")
    f.write(f"  - Precisão Weighted: {precision_weighted:.4f}\n")
    f.write(f"  - Recall Weighted: {recall_weighted:.4f}\n")
    f.write(f"  - F1-Score Weighted: {f1_weighted:.4f}\n\n")
    
    f.write("Métricas por Classe:\n")
    f.write("-" * 80 + "\n")
    for i, class_name in enumerate(class_names):
        f.write(f"\nClasse: {class_name}\n")
        f.write(f"  - Amostras: {support[i]}\n")
        f.write(f"  - Precisão: {precision[i]:.4f}\n")
        f.write(f"  - Recall (Sensibilidade): {recall[i]:.4f}\n")
        f.write(f"  - F1-Score: {f1[i]:.4f}\n")
        f.write(f"  - AUC-ROC: {auc_scores[i]:.4f}\n")
    
    f.write("\n" + "=" * 80 + "\n")
    f.write("4. RELATÓRIO DE CLASSIFICAÇÃO DETALHADO\n")
    f.write("=" * 80 + "\n\n")
    f.write(class_report)
    f.write("\n\n")
    
    f.write("=" * 80 + "\n")
    f.write("5. MATRIZ DE CONFUSÃO\n")
    f.write("=" * 80 + "\n\n")
    f.write("Legenda: Linhas = Classes Verdadeiras, Colunas = Classes Preditas\n\n")
    
    f.write("         ")
    for name in class_names:
        f.write(f"{name[:12]:>12} ")
    f.write("\n")
    
    for i, true_class in enumerate(class_names):
        f.write(f"{true_class[:12]:>12} ")
        for j in range(len(class_names)):
            f.write(f"{cm[i][j]:>12} ")
        f.write("\n")
    
    f.write("\n" + "=" * 80 + "\n")
    f.write("6. ANÁLISE DA MATRIZ DE CONFUSÃO\n")
    f.write("=" * 80 + "\n\n")
    
    for i, class_name in enumerate(class_names):
        total = np.sum(cm[i])
        correct = cm[i][i]
        incorrect = total - correct
        f.write(f"\nClasse: {class_name}\n")
        f.write(f"  - Predições corretas: {correct}/{total} ({100*correct/total:.2f}%)\n")
        f.write(f"  - Predições incorretas: {incorrect}/{total} ({100*incorrect/total:.2f}%)\n")
        if incorrect > 0:
            f.write(f"  - Confusões principais:\n")
            for j, other_class in enumerate(class_names):
                if i != j and cm[i][j] > 0:
                    f.write(f"    * Confundido com {other_class}: {cm[i][j]} vezes ({100*cm[i][j]/total:.2f}%)\n")
    
    f.write("\n" + "=" * 80 + "\n")
    f.write("7. HISTÓRICO DE TREINAMENTO\n")
    f.write("=" * 80 + "\n\n")
    
    f.write("Fase 1 - Treinamento Inicial:\n")
    f.write(f"  - Épocas executadas: {len(history1.history['loss'])}\n")
    f.write(f"  - Acurácia inicial (treino): {history1.history['accuracy'][0]:.4f}\n")
    f.write(f"  - Acurácia final (treino): {history1.history['accuracy'][-1]:.4f}\n")
    f.write(f"  - Acurácia inicial (validação): {history1.history['val_accuracy'][0]:.4f}\n")
    f.write(f"  - Acurácia final (validação): {history1.history['val_accuracy'][-1]:.4f}\n")
    f.write(f"  - Loss inicial (treino): {history1.history['loss'][0]:.4f}\n")
    f.write(f"  - Loss final (treino): {history1.history['loss'][-1]:.4f}\n")
    f.write(f"  - Loss inicial (validação): {history1.history['val_loss'][0]:.4f}\n")
    f.write(f"  - Loss final (validação): {history1.history['val_loss'][-1]:.4f}\n")
    f.write(f"  - Melhor acurácia (validação): {max(history1.history['val_accuracy']):.4f} (época {np.argmax(history1.history['val_accuracy'])+1})\n")
    f.write(f"  - Menor loss (validação): {min(history1.history['val_loss']):.4f} (época {np.argmin(history1.history['val_loss'])+1})\n\n")
    
    if history2:
        f.write("Fase 2 - Fine-tuning:\n")
        f.write(f"  - Épocas executadas: {len(history2.history['loss'])}\n")
        f.write(f"  - Acurácia inicial (treino): {history2.history['accuracy'][0]:.4f}\n")
        f.write(f"  - Acurácia final (treino): {history2.history['accuracy'][-1]:.4f}\n")
        f.write(f"  - Acurácia inicial (validação): {history2.history['val_accuracy'][0]:.4f}\n")
        f.write(f"  - Acurácia final (validação): {history2.history['val_accuracy'][-1]:.4f}\n")
        f.write(f"  - Loss inicial (treino): {history2.history['loss'][0]:.4f}\n")
        f.write(f"  - Loss final (treino): {history2.history['loss'][-1]:.4f}\n")
        f.write(f"  - Loss inicial (validação): {history2.history['val_loss'][0]:.4f}\n")
        f.write(f"  - Loss final (validação): {history2.history['val_loss'][-1]:.4f}\n")
        f.write(f"  - Melhor acurácia (validação): {max(history2.history['val_accuracy']):.4f} (época {np.argmax(history2.history['val_accuracy'])+1})\n")
        f.write(f"  - Menor loss (validação): {min(history2.history['val_loss']):.4f} (época {np.argmin(history2.history['val_loss'])+1})\n")
    
    f.write("\n" + "=" * 80 + "\n")
    f.write("8. INFORMAÇÕES ADICIONAIS\n")
    f.write("=" * 80 + "\n\n")
    f.write(f"Modelo salvo em: ./model/modelo_mobilenetv2_128x128_finetuned.keras\n")
    f.write(f"Framework: TensorFlow/Keras\n")
    f.write(f"Função de Perda: Categorical Crossentropy\n")
    f.write(f"Métrica de Avaliação: Accuracy\n\n")
    
    f.write("=" * 80 + "\n")
    f.write("FIM DO RELATÓRIO\n")
    f.write("=" * 80 + "\n")

print(f"\nRelatório detalhado salvo em: {report_path}")

cmd = ConfusionMatrixDisplay(cm, display_labels=class_names)
fig, ax = plt.subplots(figsize=(10, 8))
cmd.plot(cmap=plt.cm.Blues, ax=ax)
plt.title("Matriz de Confusão - Conjunto de Validação", fontsize=14, fontweight='bold')
plt.tight_layout()
fname = os.path.join('docs', 'confusion_matrix.png')
plt.savefig(fname, dpi=300, bbox_inches='tight')
plt.close()
print(f"Matriz de confusão salva em: {fname}")

cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
cmd_norm = ConfusionMatrixDisplay(cm_normalized, display_labels=class_names)
fig, ax = plt.subplots(figsize=(10, 8))
cmd_norm.plot(cmap=plt.cm.Blues, ax=ax, values_format='.2%')
plt.title("Matriz de Confusão Normalizada (%)", fontsize=14, fontweight='bold')
plt.tight_layout()
fname_norm = os.path.join('docs', 'confusion_matrix_normalized.png')
plt.savefig(fname_norm, dpi=300, bbox_inches='tight')
plt.close()
print(f"Matriz de confusão normalizada salva em: {fname_norm}")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].bar(class_names, precision, color='steelblue', alpha=0.8)
axes[0, 0].set_title('Precisão por Classe', fontweight='bold')
axes[0, 0].set_ylabel('Precisão')
axes[0, 0].set_ylim([0, 1.1])
axes[0, 0].grid(axis='y', alpha=0.3)
for i, v in enumerate(precision):
    axes[0, 0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=9)

axes[0, 1].bar(class_names, recall, color='coral', alpha=0.8)
axes[0, 1].set_title('Recall por Classe', fontweight='bold')
axes[0, 1].set_ylabel('Recall')
axes[0, 1].set_ylim([0, 1.1])
axes[0, 1].grid(axis='y', alpha=0.3)
for i, v in enumerate(recall):
    axes[0, 1].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=9)

axes[1, 0].bar(class_names, f1, color='seagreen', alpha=0.8)
axes[1, 0].set_title('F1-Score por Classe', fontweight='bold')
axes[1, 0].set_ylabel('F1-Score')
axes[1, 0].set_ylim([0, 1.1])
axes[1, 0].grid(axis='y', alpha=0.3)
for i, v in enumerate(f1):
    axes[1, 0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=9)

axes[1, 1].bar(class_names, auc_scores, color='mediumpurple', alpha=0.8)
axes[1, 1].set_title('AUC-ROC por Classe', fontweight='bold')
axes[1, 1].set_ylabel('AUC-ROC')
axes[1, 1].set_ylim([0, 1.1])
axes[1, 1].grid(axis='y', alpha=0.3)
for i, v in enumerate(auc_scores):
    axes[1, 1].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=9)

plt.tight_layout()
fname_metrics = os.path.join('docs', 'metrics_by_class.png')
plt.savefig(fname_metrics, dpi=300, bbox_inches='tight')
plt.close()
print(f"Métricas por classe salvas em: {fname_metrics}")

confidence_scores = np.max(pred_probs, axis=1)
plt.figure(figsize=(10, 6))
plt.hist(confidence_scores, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
plt.axvline(np.mean(confidence_scores), color='red', linestyle='--', linewidth=2, label=f'Média: {np.mean(confidence_scores):.3f}')
plt.axvline(np.median(confidence_scores), color='green', linestyle='--', linewidth=2, label=f'Mediana: {np.median(confidence_scores):.3f}')
plt.title('Distribuição de Confiança das Predições', fontsize=14, fontweight='bold')
plt.xlabel('Confiança (Probabilidade Máxima)')
plt.ylabel('Frequência')
plt.legend()
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
fname_conf = os.path.join('docs', 'prediction_confidence_distribution.png')
plt.savefig(fname_conf, dpi=300, bbox_inches='tight')
plt.close()
print(f"Distribuição de confiança salva em: {fname_conf}")

from sklearn.metrics import roc_curve, auc
plt.figure(figsize=(12, 8))
colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']

for i, class_name in enumerate(class_names):
    fpr, tpr, _ = roc_curve(true_labels_bin[:, i], pred_probs[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color=colors[i % len(colors)], lw=2, 
             label=f'{class_name} (AUC = {roc_auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Chance (AUC = 0.5)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falsos Positivos', fontsize=12)
plt.ylabel('Taxa de Verdadeiros Positivos', fontsize=12)
plt.title('Curvas ROC - Multi-classe (One-vs-Rest)', fontsize=14, fontweight='bold')
plt.legend(loc='lower right')
plt.grid(alpha=0.3)
plt.tight_layout()
fname_roc = os.path.join('docs', 'roc_curves.png')
plt.savefig(fname_roc, dpi=300, bbox_inches='tight')
plt.close()
print(f"Curvas ROC salvas em: {fname_roc}")

os.makedirs('./model', exist_ok=True)
model.save("./model/modelo_mobilenetv2_128x128_finetuned.keras")

def plot_history(histories, labels, out_dir='./docs'):
    os.makedirs(out_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    for hist, label in zip(histories, labels):
        epochs_offset = 0 if label == 'Inicial' else len(histories[0].history['accuracy'])
        epochs = range(epochs_offset + 1, epochs_offset + len(hist.history['accuracy']) + 1)
        plt.plot(epochs, hist.history['accuracy'], label=f'{label} - Treino', linewidth=2)
        plt.plot(epochs, hist.history['val_accuracy'], label=f'{label} - Validação', linestyle='--', linewidth=2)
    
    plt.title('Acurácia durante o Treinamento', fontsize=14, fontweight='bold')
    plt.xlabel('Épocas', fontsize=12)
    plt.ylabel('Acurácia', fontsize=12)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    for hist, label in zip(histories, labels):
        epochs_offset = 0 if label == 'Inicial' else len(histories[0].history['loss'])
        epochs = range(epochs_offset + 1, epochs_offset + len(hist.history['loss']) + 1)
        plt.plot(epochs, hist.history['loss'], label=f'{label} - Treino', linewidth=2)
        plt.plot(epochs, hist.history['val_loss'], label=f'{label} - Validação', linestyle='--', linewidth=2)
    
    plt.title('Loss durante o Treinamento', fontsize=14, fontweight='bold')
    plt.xlabel('Épocas', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fname = os.path.join(out_dir, 'training_history.png')
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Histórico de treinamento salvo em: {fname}")
    
    plt.figure(figsize=(10, 6))
    for hist, label in zip(histories, labels):
        epochs_offset = 0 if label == 'Inicial' else len(histories[0].history['accuracy'])
        epochs = range(epochs_offset + 1, epochs_offset + len(hist.history['accuracy']) + 1)
        plt.plot(epochs, hist.history['accuracy'], label=f'{label} - Treino', linewidth=2, marker='o', markersize=3)
        plt.plot(epochs, hist.history['val_accuracy'], label=f'{label} - Validação', linestyle='--', linewidth=2, marker='s', markersize=3)
    
    plt.title('Evolução da Acurácia', fontsize=14, fontweight='bold')
    plt.xlabel('Épocas', fontsize=12)
    plt.ylabel('Acurácia', fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fname_acc = os.path.join(out_dir, 'accuracy_evolution.png')
    plt.savefig(fname_acc, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Evolução da acurácia salva em: {fname_acc}")
    
    plt.figure(figsize=(10, 6))
    for hist, label in zip(histories, labels):
        epochs_offset = 0 if label == 'Inicial' else len(histories[0].history['loss'])
        epochs = range(epochs_offset + 1, epochs_offset + len(hist.history['loss']) + 1)
        plt.plot(epochs, hist.history['loss'], label=f'{label} - Treino', linewidth=2, marker='o', markersize=3)
        plt.plot(epochs, hist.history['val_loss'], label=f'{label} - Validação', linestyle='--', linewidth=2, marker='s', markersize=3)
    
    plt.title('Evolução do Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Épocas', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fname_loss = os.path.join(out_dir, 'loss_evolution.png')
    plt.savefig(fname_loss, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Evolução do loss salva em: {fname_loss}")
    
    plt.figure(figsize=(10, 6))
    for hist, label in zip(histories, labels):
        epochs_offset = 0 if label == 'Inicial' else len(histories[0].history['accuracy'])
        epochs = range(epochs_offset + 1, epochs_offset + len(hist.history['accuracy']) + 1)
        acc_diff = np.array(hist.history['accuracy']) - np.array(hist.history['val_accuracy'])
        plt.plot(epochs, acc_diff, label=f'{label}', linewidth=2, marker='o', markersize=3)
    
    plt.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.3)
    plt.title('Gap de Acurácia (Treino - Validação)\nIndicador de Overfitting', fontsize=14, fontweight='bold')
    plt.xlabel('Épocas', fontsize=12)
    plt.ylabel('Diferença de Acurácia', fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fname_gap = os.path.join(out_dir, 'overfitting_analysis.png')
    plt.savefig(fname_gap, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Análise de overfitting salva em: {fname_gap}")

if history2:
    plot_history([history1, history2], ['Inicial', 'Fine-tuning'])
else:
    plot_history([history1], ['Inicial'])

print("\n" + "=" * 80)
print("TREINAMENTO CONCLUÍDO COM SUCESSO!")
print("=" * 80)
print(f"Modelo salvo em: ./model/modelo_mobilenetv2_128x128_finetuned.keras")
print(f"Relatório completo e gráficos salvos em: ./docs/")
print("=" * 80 + "\n")