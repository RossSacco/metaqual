import pyterrier_pisa
import pyterrier_dr

class RetrievalPipelines:
    def __init__(self, index_path, query_encoder=None):
        """
        index_path: Percorso all'indice
        query_encoder: Il modello caricato su GPU (opzionale, usato per SPLADE e TAS-B)
        """
        self.index_path = index_path
        self.query_encoder = query_encoder

    def get_bm25(self):
        """Pipeline BM25 tramite PISA."""
        return pyterrier_pisa.PisaIndex(self.index_path).bm25()

    def get_splade(self):
        """Pipeline SPLADE v2 tramite PISA."""
        if self.query_encoder is None:
            raise ValueError("Devi passare il query_encoder per SPLADE!")
        # AGGIUNTO stemmer="none"
        return self.query_encoder >> pyterrier_pisa.PisaIndex(self.index_path, stemmer="none").quantized()

    def get_tasb(self):
        if self.query_encoder is None:
            raise ValueError("Devi passare il modello/encoder per TAS-B!")
        
        if self.index_path.endswith(".flex"):
        # self.query_encoder ora contiene il modello TasB.dot()
        # La concatenazione >> imposta automaticamente la metrica corretta
            return self.query_encoder >> pyterrier_dr.FlexIndex(self.index_path)
            
        #if self.index_path.endswith(".flex"):
        #    return self.query_encoder >> pyterrier_dr.FlexIndex(self.index_path).dot()
        # metaqual/retrieval/pyterrier_pipe.py


        
        