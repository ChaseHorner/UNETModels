import matplotlib.pyplot as plt

def chart_metrics(metrics, model_folder, EPOCHS):
    plt.figure(figsize=(30, 10))

    # PSNR
    plt.subplot(1, 3, 1)
    plt.plot(range(1, EPOCHS + 1), metrics["train_psnr"], label='PSNR Training', color='blue', linestyle='--')
    plt.plot(range(1, EPOCHS + 1), metrics["valid_psnr"], label='PSNR Validation', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('PSNR')
    plt.legend()
    plt.title('PSNR')

    # SSIM
    plt.subplot(1, 3, 2)
    plt.plot(range(1, EPOCHS + 1), metrics["train_ssim"], label='SSIM Training', color='red', linestyle='--')
    plt.plot(range(1, EPOCHS + 1), metrics["valid_ssim"], label='SSIM Validation', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('SSIM')
    plt.legend()
    plt.title('SSIM')

    # Loss
    plt.subplot(1, 3, 3)
    plt.plot(range(1, EPOCHS + 1), metrics["train_loss"], label='Loss Training', color='green', linestyle='--')
    plt.plot(range(1, EPOCHS + 1), metrics["valid_loss"], label='Loss Validation', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss')

    plt.tight_layout()
    plt.savefig(model_folder + '/psnr_ssim_loss.png')