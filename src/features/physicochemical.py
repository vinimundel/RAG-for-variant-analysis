class PhysicochemicalEngine:
    """Compute biochemical deltas for one wild-type-to-mutant substitution."""

    # Electrostatic Charge at physiological pH
    CHARGE = {
        'R': 1.0, 'K': 1.0, 'H': 0.1,
        'D': -1.0, 'E': -1.0,
        'A': 0.0, 'N': 0.0, 'C': 0.0, 'Q': 0.0, 'G': 0.0, 'I': 0.0, 'L': 0.0, 'M': 0.0,
        'F': 0.0, 'P': 0.0, 'S': 0.0, 'T': 0.0, 'W': 0.0, 'Y': 0.0, 'V': 0.0
    }

    # Kyte-Doolittle Hydropathy Index (Positive = Hydrophobic)
    HYDROPHOBICITY = {
        'I': 4.5, 'V': 4.2, 'L': 3.8, 'F': 2.8, 'C': 2.5, 'M': 1.9, 'A': 1.8,
        'G': -0.4, 'T': -0.7, 'S': -0.8, 'W': -0.9, 'Y': -1.3, 'P': -1.6,
        'H': -3.2, 'E': -3.5, 'Q': -3.5, 'D': -3.5, 'N': -3.5, 'K': -3.9, 'R': -4.5
    }

    # Van der Waals Volume (Angstroms^3)
    VOLUME = {
        'G': 60.1, 'A': 88.6, 'S': 89.0, 'C': 108.5, 'D': 111.1, 'P': 112.7,
        'N': 114.1, 'T': 116.1, 'E': 138.4, 'V': 140.0, 'Q': 143.8, 'H': 153.2,
        'M': 162.9, 'I': 166.7, 'L': 166.7, 'K': 168.6, 'R': 173.4, 'F': 189.9,
        'Y': 193.6, 'W': 227.8
    }

    @classmethod
    def get_deltas(cls, wt_aa: str, mut_aa: str) -> dict:
        """Calculates absolute change in parameters: Mutant - WT."""
        wt_aa = wt_aa.upper()
        mut_aa = mut_aa.upper()

        if wt_aa not in cls.CHARGE or mut_aa not in cls.CHARGE:
            return {"delta_charge": 0.0, "delta_hydrophobicity": 0.0, "delta_volume": 0.0}

        return {
            "delta_charge": float(cls.CHARGE[mut_aa] - cls.CHARGE[wt_aa]),
            "delta_hydrophobicity": float(cls.HYDROPHOBICITY[mut_aa] - cls.HYDROPHOBICITY[wt_aa]),
            "delta_volume": float(cls.VOLUME[mut_aa] - cls.VOLUME[wt_aa])
        }
