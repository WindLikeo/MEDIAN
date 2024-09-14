import torch



# making the sequence mask
def make_mask(feature):
    """
    :param feature:
        for img: (bs, proposals, embedding_size)
        for text - do text.unsqueeze(2) first : (bs, seq_len, 1)
    :return:
        shape: (bs, 1, 1, seq_len/proposal)
    """
    return (torch.sum(
        torch.abs(feature),
        dim=-1
    ) == 0).unsqueeze(1).unsqueeze(2)


if __name__ == '__main__':
    print("make mask")
