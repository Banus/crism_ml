"""Manipulation of laboratory spectra and labels.

We reserve a mineral label 'artifact' for an artifact [1]_ generated by the
convolution of the atmospheric filter with noise spikes during CRISM I/F data
preprocessing.

.. rubric:: References

.. [1] Leask, E., Ehlmann, B., Dundar, M., Murchie, S., & Seelos, F. (2018).
  Challenges in the search for perchlorate and other hydrated minerals with
  2.1-µm absorptions on Mars. Geophysical Research Letters, 45(22), 12–180.
"""

# detailed categories - some classes are hard to discriminate
FULL_NAMES = {
    1: 'CO2 Ice',
    2: 'H2O Ice',
    3: 'Gypsum',
    4: 'Ferric Hydroxysulfate',
    5: "Hematite",
    6: 'Nontronite',
    7: 'Saponite',
    8: 'Prehnite',  # Prehnite Zeolite
    9: 'Jarosite',
    10: 'Serpentine',
    11: 'Alunite',
    12: 'Akaganeite',
    13: 'Ca/Fe CO3',  # Calcite, Ca/Fe carbonate
    14: 'Beidellite',
    15: 'Kaolinite',
    16: 'Bassanite',
    17: 'Epidote',
    18: 'Montmorillonite',
    19: 'Rosenite',
    20: 'Mg Cl salt',  # Mg(ClO3)2.6H2O
    21: 'Halloysite',
    22: 'Epsomite',
    23: 'Illite/Muscovite',
    24: 'Margarite',
    25: 'Analcime',  # Zeolite
    26: 'Monohydrated sulfate',  # Szomolnokite
    27: 'Opal 1',  # Opal
    28: 'Opal 2',  # Opal-A
    29: 'Iron Oxide Silicate Sulfate',
    30: 'MgCO3',  # Magnesite
    31: 'Chlorite',
    32: 'Clinochlore',
    33: 'Low Ca Pyroxene',
    34: 'Olivine Forsterite',
    35: 'High Ca Pyroxene',
    36: 'Olivine Fayalite',
    37: 'Chloride',
    38: 'Artifact',
    39: 'Neutral',
    40: 'Feldspar Plagioclase',  # not in the dataset
}

FLAT_CATS = list(range(33, 41))

# broad categories
BROAD_NAMES = {
    **FULL_NAMES,
    5: 'Nondiagnostic', 6: 'Fe smectite', 7: 'Mg smectite',
    14: 'Al smectite', 18: 'Al smectite', 19: 'Polyhydrated sulfate',
    20: 'Polyhydrated sulfate', 21: 'Kaolinite', 22: 'Polyhydrated sulfate',
    23: 'Illite', 24: 'Illite', 27: 'Hydrated silica',
    28: 'Hydrated silica', 32: 'Chlorite', 34: 'Mg Olivine',
    36: 'Fe Olivine', 37: 'Nondiagnostic', 39: "Bland"
}

# unique names for classes in the training set
CLASS_NAMES = {
    **BROAD_NAMES,
    5: "Hematite", 14: "Beidellite", 18: "Montmorillonite", 20: "Mg Cl salt",
    34: "Olivine", 37: "Chloride"
}


# similar minerals, mapped to the same label
#  halloysite -> kaolinite; epsomite -> rosenite; margarite -> illite;
#   clinochlore -> chlorite
ALIASES_LAB = {21: 15, 22: 19, 24: 23, 32: 31}
# opal 2 -> opal 1
ALIASES = {**ALIASES_LAB, 28: 27}
# Olivine Fayalite -> Olivine Forsterite
ALIASES_TRAIN = {**ALIASES, 36: 34}

# merge classes associated with the same broad names
# chloride -> hematite, montmorillonite -> beidellite, Mg Cl salt -> rosenite
ALIASES_EVAL = {**ALIASES_TRAIN, 37: 5, 18: 14, 20: 19}

ALIAS_LAB_NAMES = {19: 'Polyhydrated sulfate',
                   23: 'Illite-Muscovite-Margarite'}
ALIAS_NAMES = {**ALIAS_LAB_NAMES, 27: "Opal"}
ALIAS_TRAIN_NAME = {**ALIAS_NAMES, 34: 'Olivine'}

# mineral names that will never be detected
BAD_LABELS = ["Garnet", "Cerussite", "Andradite", "Quartz", "Labradorite",
              "Plagioclase", "Thermonatrite", "Phlogopite", "NaCl",
              "Actinolite", "Anihydrite", "Anorthite", "Topaz", "Aubrite",
              "Arsenopyrite", "MgO", "Wollastonite", "Troillite", "Amphibole",
              "Feldspar"]


def relabel(pixlabs, aliases=None):
    """Merge label belonging to the same category of minerals.

    Parameters
    __________
    pixlabs: ndarray
        array of labels to merge
    aliases: dict
        dictionary mapping a class to be replaced to the target class; if None,
        'ALIASES' is used by default.

    Returns
    -------
    pixlabs: ndarray
        the remapped labels; in-place processing
    """
    if aliases is None:
        aliases = ALIASES

    for lbl, dst in aliases.items():
        pixlabs[pixlabs == lbl] = dst
    return pixlabs


if __name__ == '__main__':
    pass