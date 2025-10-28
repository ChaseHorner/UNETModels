import matplotlib.pyplot as plt

def chart_metrics(metrics, model_folder, EPOCHS):
    plt.figure(figsize=(30, 20))

    # MSE
    plt.subplot(3, 2, 1)
    plt.plot(range(1, EPOCHS + 1), metrics["train_mses"], label='MSE Training', color='blue', linestyle='--')
    plt.plot(range(1, EPOCHS + 1), metrics["eval_mses"], label='MSE Validation', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.legend()
    plt.title('MSE')

    # RMSE
    plt.subplot(3, 2, 2)
    plt.plot(range(1, EPOCHS + 1), metrics["train_rmses"], label='RMSE Training', color='red', linestyle='--')
    plt.plot(range(1, EPOCHS + 1), metrics["eval_rmses"], label='RMSE Validation', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('RMSE')
    plt.legend()
    plt.title('RMSE')

    # MAE
    plt.subplot(3, 2, 3)
    plt.plot(range(1, EPOCHS + 1), metrics["train_maes"], label='MAE Training', color='green', linestyle='--')
    plt.plot(range(1, EPOCHS + 1), metrics["eval_maes"], label='MAE Validation', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()
    plt.title('MAE')

    # SSIM
    plt.subplot(3, 2, 4)
    plt.plot(range(1, EPOCHS + 1), metrics["train_ssims"], label='SSIM Training', color='orange', linestyle='--')
    plt.plot(range(1, EPOCHS + 1), metrics["eval_ssims"], label='SSIM Validation', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('SSIM')
    plt.legend()
    plt.title('SSIM')

    # Field Difference
    plt.subplot(3, 2, 5)
    plt.plot(range(1, EPOCHS + 1), metrics["train_field_diffs"], label='Field Diff Training', color='purple', linestyle='--')
    plt.plot(range(1, EPOCHS + 1), metrics["eval_field_diffs"], label='Field Diff Validation', color='purple')
    plt.xlabel('Epochs')
    plt.ylabel('Field Difference')
    plt.legend()
    plt.title('Field Difference')

    plt.tight_layout()
    plt.savefig(model_folder + '/loss.png')


if __name__ == "__main__":
    import json
    MODEL_FOLDER = 'outputs/UNET_v1.2.1'  # Change to the desired model folder
    model_status_path = f'{MODEL_FOLDER}/status.json'
    with open(model_status_path, "r") as f:
        model_status = json.load(f)

    chart_metrics(model_status["metrics"], MODEL_FOLDER, model_status["last_trained_epoch"])