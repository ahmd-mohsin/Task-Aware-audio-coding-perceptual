from data_loaders import RealisticAudioDataset
# if __name__ == '__main__':
#     """python -m data_loaders.realistic_audio"""
dset = RealisticAudioDataset(
    dataset='val',  #, cv_dev93, test_eval92
    audio_time_len=None,
    dataset_dir='../real_man_dataset',
    noise_dir=None,
    record_dir='../real_man_dataset/val/ma_noisy_speech',
    target_dir='../real_man_dataset/val/dp_speech',
    spk_pattern='all',
    noise_type='real',
    use_microphone_array_generalization=True,
)


print(dset.length)
for i in range(dset.length):
    one_item = dset.__getitem__((i, i))
    print(one_item)
    break 


# for i in one_item:
#     print(i.shape)