from aletheia.scripts.extract_features import FEATURE_EXTRACTORS

for k, class_ in FEATURE_EXTRACTORS.items():
    f = class_()
    print(k, f.model.num_parameters())