import matplotlib.pyplot as plt 

def plot_train_validation_curve(tl_list, vl_list, tp_list, vp_list, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 5))

    ax1.plot(tl_list, label='Train Loss', marker='o')
    ax1.plot(vl_list, label='Valid Loss', marker='s')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss', fontweight='bold')
    ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(tp_list, label='Train PPL', marker='o')
    ax2.plot(vp_list, label='Valid PPL', marker='s')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Perplexity')
    ax2.set_title('Training and Validation Perplexity', fontweight='bold')
    ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path + '/training_curves.png', dpi=150, bbox_inches='tight')
    plt.show()
   