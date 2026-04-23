# metaqual/data/loaders/dataset_loader.py
import pyterrier as pt

class DatasetLoader:
    """
    Gestisce il caricamento dinamico di un dataset PyTerrier.
    """
    def __init__(self, dataset_name="msmarco_passage"):
        if not pt.started():
            pt.init()
            
        print(f"Inizializzazione loader per il dataset '{dataset_name}'...")
        self.dataset_name = dataset_name
        self.dataset = pt.get_dataset(dataset_name)

    def get_corpus_iter(self):
        return self.dataset.get_corpus_iter()

    def get_topics(self, variant="test"):
        return self.dataset.get_topics(variant)

    def get_qrels(self, variant="test"):
        return self.dataset.get_qrels(variant)

    def get_prebuilt_index(self, index_type):
        print(f"Recupero indice pre-costruito ({index_type}) per {self.dataset_name}...")
        return self.dataset.get_index(index_type)