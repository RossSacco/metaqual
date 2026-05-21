import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def plot_roc(input_file="roc_data.pkl"):
    with open(input_file, "rb") as f:
        data = pickle.load(f)
    
    labels = data["labels"]
    
    fig, ax = plt.subplots(figsize=(8, 8))
    axins = ax.inset_axes([0.05, 0.7, 0.25, 0.25])
    
    colors = {
        'metadata_qualt5_nuovo': '#9467bd',
        'finetuned_qualt5': '#1f77b4',
        'metadata_qualt5': '#ff7f0e',
        'itn': '#d62728',
        'cdd': '#d62728',
        'perplexity': '#e377c2',
        'tasb': '#2ca02c',
    }

    legend_names = {
        'finetuned_qualt5': 'QualT5-Finetuned',
        'metadata_qualt5': 'Metadata-QualT5',
        'metadata_qualt5_nuovo': 'Metadata-QualT5-Nuovo',
        'perplexity': 'T5-Ppl',
        'tasb': 'TASB-Mag',
        'itn': 'ITN',
        'cdd': 'CDD',
    }

    for name, scores in data["scorers"].items():
        fpr, tpr, _ = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)
        
        linestyle = '--' if name in ['cdd', 'perplexity'] else '-'
        color = colors.get(name, 'black')
        
        display_name = legend_names.get(name, name.upper())
        label_text = f'{display_name} ({roc_auc:.2f})'
        
        ax.plot(fpr, tpr, label=label_text, color=color, linestyle=linestyle)
        axins.plot(fpr, tpr, color=color, linestyle=linestyle)

    ax.plot([0, 1], [0, 1], color='black', linestyle=':', label='Random (0.50)')
    axins.plot([0, 1], [0, 1], color='black', linestyle=':')

    ax.set_xlim([0.8, 1.0])
    ax.set_ylim([0.8, 1.0])
    ax.set_xlabel('False Positive Rate\n(higher -> more non-relevant passages kept)')
    ax.set_ylabel('True Positive Rate\n(higher -> fewer relevant passages pruned)')
    ax.set_title('ROC Curves for Passage Quality Estimators')
    ax.legend(loc="lower right")
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    
    axins.set_xlim([0.0, 1.0])
    axins.set_ylim([0.0, 1.0])
    axins.set_xticks([0, 1])
    axins.set_yticks([0, 1])
    axins.set_title("detail", fontsize=10, loc='left', pad=-12)
    
    ax.indicate_inset_zoom(axins, edgecolor="black")

    output_path = "metaqual/utils/plot/roc_metaqual2.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"Grafico salvato in: {output_path}")

if __name__ == "__main__":
    plot_roc()