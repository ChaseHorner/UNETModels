import matplotlib.pyplot as plt

def chart_metrics(metrics, model_folder, EPOCHS):
    plt.figure(figsize=(40, 10))

    # MSE
    plt.subplot(1, 4, 1)
    plt.plot(range(1, EPOCHS + 1), metrics["train_MSE"], label='MSE Training', color='blue', linestyle='--')
    plt.plot(range(1, EPOCHS + 1), metrics["eval_MSE"], label='MSE Validation', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.legend()
    plt.title('MSE')

    # RMSE
    plt.subplot(1, 4, 2)
    plt.plot(range(1, EPOCHS + 1), metrics["train_rmse"], label='RMSE Training', color='red', linestyle='--')
    plt.plot(range(1, EPOCHS + 1), metrics["eval_rmse"], label='RMSE Validation', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('RMSE')
    plt.legend()
    plt.title('RMSE')

    # MAE
    plt.subplot(1, 4, 3)
    plt.plot(range(1, EPOCHS + 1), metrics["train_mae"], label='MAE Training', color='green', linestyle='--')
    plt.plot(range(1, EPOCHS + 1), metrics["eval_mae"], label='MAE Validation', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()
    plt.title('MAE')

    # SSIM
    plt.subplot(1, 4, 4)
    plt.plot(range(1, EPOCHS + 1), metrics["train_ssim"], label='SSIM Training', color='orange', linestyle='--')
    plt.plot(range(1, EPOCHS + 1), metrics["eval_ssim"], label='SSIM Validation', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('SSIM')
    plt.legend()
    plt.title('SSIM')

    plt.tight_layout()
    plt.savefig(model_folder + '/loss.png')