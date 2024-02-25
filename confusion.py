import numpy as np
import matplotlib.pyplot as plt

# Matrices de confusión de ejemplo
confusion_matrix0 = np.array([[246, 33], [68, 182]])
confusion_matrix1 = np.array([[247, 32], [58, 192]])
confusion_matrix2 = np.array([[240, 39], [110, 140]])
confusion_matrix3 = np.array([[264, 15], [70, 180]])

# Configuración de la cuadrícula y los gráficos
fig, axs = plt.subplots(2, 2, figsize=(10, 10))

# Configuración de los gráficos individuales
for i in range(2):
    for j in range(2):
        axs[i, j].imshow(eval(f'confusion_matrix{i*2+j}'), cmap='Blues')

        axs[i, j].set_xticks([0, 1])
        axs[i, j].set_yticks([0, 1])
        axs[i, j].set_xticklabels(['Control', 'TDAH'])
        axs[i, j].set_yticklabels(['Control', 'TDAH'])

        for x in range(2):
            for y in range(2):
                axs[i, j].text(y, x, str(eval(f'confusion_matrix{i*2+j}[x, y]')), color='grey', ha='center', va='center', fontsize=20)

axs[0, 0].set_title('TabNet')
axs[0, 0].set_ylabel('Valores reales')
axs[0, 0].set_xlabel('Valores predichos')

axs[0, 1].set_title('TabTransformer')
axs[0, 1].set_ylabel('Valores reales')
axs[0, 1].set_xlabel('Valores predichos')

axs[1, 0].set_title('Node')
axs[1, 0].set_ylabel('Valores reales')
axs[1, 0].set_xlabel('Valores predichos')

axs[1, 1].set_title('1DCNN')
axs[1, 1].set_ylabel('Valores reales')
axs[1, 1].set_xlabel('Valores predichos')

plt.tight_layout()
plt.show()
