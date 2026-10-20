"""Final training recipe; this entire file is embedded in train_models.ipynb."""
import gc
import time
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


def fit_final_seed(images, labels, seed, configuration):
    """Retrain every learned component from scratch, with no development caches."""
    members = []
    timings = []
    # Within one seed only: identical dictionary recipes share exact features.
    # This cache is rebuilt from raw training data on every Run All and seed.
    dictionary_cache = {}
    feature_cache = {}
    key_of = lambda s: (s['clusters'], s.get('patch_size', 6), s.get('n_patches', 60000),
                        s['smooth_sigma'], s.get('preprocess'))
    keys = [key_of(specification) for specification in configuration['members']]
    for index, specification in enumerate(configuration['members']):
        started = time.perf_counter()
        key = keys[index]
        if key not in dictionary_cache:
            print(f'seed={seed} learning dictionary {key}', flush=True)
            dictionary_cache[key] = learn_dictionary(images, seed,
                n_clusters=key[0], patch_size=key[1], n_patches=key[2], smooth_sigma=key[3],
                preprocess=key[4])
            feature_cache[key] = image_features(images, banks=[dictionary_cache[key]])
            print(f'seed={seed} original features ready: {feature_cache[key].shape}', flush=True)
        bank = dictionary_cache[key]
        original_features = feature_cache[key]
        if key not in keys[index + 1:]:
            del feature_cache[key]  # Bound peak memory; no later member needs it.
        ratio = specification['augmentation_ratio']
        rng = np.random.default_rng(seed)
        chosen = rng.choice(len(images), size=int(len(images) * ratio), replace=ratio > 1)
        augmented = augment_images(images[chosen], seed, mode=specification.get('augmentation_mode', 'robust'))
        augmented_features = image_features(augmented, banks=[bank])
        print(f'seed={seed} augmented features ready: {augmented_features.shape}', flush=True)
        del augmented
        features = np.concatenate([original_features, augmented_features])
        targets = np.concatenate([labels, labels[chosen]])
        del original_features, augmented_features
        scaler = StandardScaler(copy=False)
        features = scaler.fit_transform(features)
        classifier = MLPClassifier(hidden_layer_sizes=tuple(specification['hidden_layers']),
            alpha=specification['alpha'], learning_rate_init=specification['learning_rate'],
            batch_size=512, max_iter=1, random_state=seed, shuffle=True)
        for epoch in range(specification['epochs']):
            with threadpool_limits(limits=configuration.get('training_threads', 1)):
                classifier.partial_fit(features, targets, classes=np.arange(10))
            flush_tiny_training_values(classifier, configuration.get('tiny_value_threshold', 1e-20))
            print(f"seed={seed} K={specification['clusters']} epoch={epoch+1}/{specification['epochs']} loss={classifier.loss_:.4f}", flush=True)
        before = classifier.predict_proba(features[:37])
        strip_training_state(classifier)
        np.testing.assert_array_equal(before, classifier.predict_proba(features[:37]))
        del features, targets
        members.append(make_pipeline([bank], classifier, scaler=scaler))
        elapsed = time.perf_counter() - started
        timings.append(round(elapsed, 2))
        print(f'Member completed in {elapsed:.1f}s', flush=True)
        gc.collect()
    return ImageClassifier(members, weights=configuration['weights'],
                           horizontal_tta=configuration['horizontal_tta']), timings
