import matplotlib.pyplot as plt

def chart_metrics(metrics, model_folder, EPOCHS):
    plt.figure(figsize=(40, 10))

    # PSNR
    plt.subplot(1, 4, 1)
    plt.plot(range(1, EPOCHS + 1), metrics["train_psnr"], label='PSNR Training', color='blue', linestyle='--')
    plt.plot(range(1, EPOCHS + 1), metrics["eval_psnr"], label='PSNR Validation', color='blue')
    plt.xlabel('Epochs')
    plt.ylabel('PSNR')
    plt.legend()
    plt.title('PSNR')

    # SSIM
    plt.subplot(1, 4, 2)
    plt.plot(range(1, EPOCHS + 1), metrics["train_ssim"], label='SSIM Training', color='red', linestyle='--')
    plt.plot(range(1, EPOCHS + 1), metrics["eval_ssim"], label='SSIM Validation', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('SSIM')
    plt.legend()
    plt.title('SSIM')

    # wLoss
    plt.subplot(1, 4, 3)
    plt.plot(range(1, EPOCHS + 1), metrics["train_wloss"], label='wLoss Training', color='green', linestyle='--')
    plt.plot(range(1, EPOCHS + 1), metrics["eval_wloss"], label='wLoss Validation', color='green')
    plt.xlabel('Epochs')
    plt.ylabel('wLoss')
    plt.legend()
    plt.title('wLoss')

    # L1
    plt.subplot(1, 4, 4)
    plt.plot(range(1, EPOCHS + 1), metrics["train_l1"], label='L1 Training', color='orange', linestyle='--')
    plt.plot(range(1, EPOCHS + 1), metrics["eval_l1"], label='L1 Validation', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('L1')
    plt.legend()
    plt.title('L1')

    plt.tight_layout()
    plt.savefig(model_folder + '/psnr_ssim_loss_L1.png')