import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def plot_roc(input_file="roc_data.pkl"):
    with open(input_file, "rb") as f:
        data = pickle.load(f)
    
    labels = data["labels"]
    
    # Inizializziamo la figura e l'asse principale
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Creiamo l'asse per il "grafico nel grafico" in alto a sinistra
    axins = ax.inset_axes([0.05, 0.7, 0.25, 0.25])
    
    # Colori simili al grafico originale
    colors = {'qualt5': '#1f77b4', 'itn': '#d62728', 'cdd': '#d62728', 
              'perplexity': '#e377c2', 'tasb': '#2ca02c'}

    # --- NUOVO: Dizionario per i nomi personalizzati nella legenda ---
    legend_names = {
        'qualt5': 'QualT5-Small',
        'perplexity': 'T5-Ppl',
        'tasb': 'TASB-Mag'
    }

    for name, scores in data["scorers"].items():
        fpr, tpr, _ = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)
        
        linestyle = '--' if name in ['cdd', 'perplexity'] else '-'
        color = colors.get(name, 'black')
        
        # Recupera il nome personalizzato. Se non esiste (es. per itn o cdd), usa il nome in maiuscolo
        display_name = legend_names.get(name, name.upper())
        label_text = f'{display_name} ({roc_auc:.2f})'
        
        # 1. Disegniamo la curva sul grafico grande (che poi zoomeremo)
        ax.plot(fpr, tpr, label=label_text, color=color, linestyle=linestyle)
        
        # 2. Disegniamo la STESSA curva sul grafico piccolo (che mostrerà tutto da 0 a 1)
        axins.plot(fpr, tpr, color=color, linestyle=linestyle)

    # Baseline Random per entrambi i grafici
    ax.plot([0, 1], [0, 1], color='black', linestyle=':', label='Random (0.50)')
    axins.plot([0, 1], [0, 1], color='black', linestyle=':')

    # --- Configurazione Grafico Principale (Zoomato) ---
    ax.set_xlim([0.8, 1.0])
    ax.set_ylim([0.8, 1.0])
    ax.set_xlabel('False Positive Rate\n(higher -> more non-relevant passages kept)')
    ax.set_ylabel('True Positive Rate\n(higher -> fewer relevant passages pruned)')
    ax.set_title('ROC Curves for Passage Quality Estimators')
    ax.legend(loc="lower right")
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    
    # --- Configurazione Grafico Piccolo (Dettaglio Globale) ---
    axins.set_xlim([0.0, 1.0])
    axins.set_ylim([0.0, 1.0])
    axins.set_xticks([0, 1])
    axins.set_yticks([0, 1])
    axins.set_title("detail", fontsize=10, loc='left', pad=-12) 
    
    # Disegna il rettangolo che fa capire da dove arriva lo zoom
    ax.indicate_inset_zoom(axins, edgecolor="black")

    # Salvataggio
    output_path = "metaqual/utils/plot/roc_reproduction.png"
    plt.savefig(output_path, dpi=300)
    print(f"Grafico salvato in: {output_path}")

if __name__ == "__main__":
    plot_roc()