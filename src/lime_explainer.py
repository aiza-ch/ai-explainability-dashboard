import lime
import lime.lime_tabular

def explain_instance_with_lime(model, X_train_scaled, X_test_scaled, feature_names, index=0):
    # Create LIME explainer
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train_scaled,
        feature_names=feature_names,
        class_names=['No Default', 'Default'],
        mode='classification'
    )

    # Generate explanation for the selected instance
    explanation = explainer.explain_instance(
        data_row=X_test_scaled[index],
        predict_fn=model.predict_proba
    )

    return explanation.as_list()
