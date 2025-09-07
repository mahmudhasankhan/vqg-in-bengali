"""Loads question answering data and feeds it to the models.
"""

import copy
import json
import os
import pickle
import h5py
import numpy as np
import torch
import torch.utils.data as data

if os.path.exists("vocab.pkl"):
    vocab = pickle.load(open("vocab.pkl", "rb"))
else:
    import utils.vocab
    print("Building Vocab")
    vocab = utils.vocab.build_vocab('data/vqa/Bangla/Bangla_Train_Ques_220K.json',
                                    'data/vqa/Bangla/translated_iq_dataset.json', 4)


class IQDataset(data.Dataset):
    """Custom Dataset compatible with torch.utils.data.DataLoader.
    """

    def __init__(self, dataset, transform=None, max_examples=None,
                 indices=None):
        """Set the path for images, captions and vocabulary wrapper.

        Args:
            dataset: hdf5 file with questions and images.
            images: hdf5 file with questions and imags.
            transform: image transformer.
            max_examples: Used for debugging. Assumes that we have a
                maximum number of training examples.
            indices: List of indices to use.
        """
        self.dataset = dataset
        self.transform = transform
        self.max_examples = max_examples
        self.indices = indices
        self.max_len = 23

        self.cat2name = sorted(
            json.load(open("data/processed/Bangla/cat2name.json", "r")))

    def __getitem__(self, index):
        """Returns one data pair (image and caption).
        """
        if not hasattr(self, 'images'):
            annos = h5py.File(self.dataset, 'r')
            self.questions = annos['questions']
            self.answers = annos['answers']
            self.answer_types = annos['answer_types']
            self.image_indices = annos['image_indices']
            self.images = annos['images']
            self.image_ids = annos["image_ids"]

        if self.indices is not None:
            index = self.indices[index]

        question = self.questions[index]
        posterior = copy.deepcopy(question)
        posterior[0] = vocab.word2idx[vocab.SYM_POS]
        posterior = posterior.tolist()
        try:
            posterior.remove(vocab.word2idx[vocab.SYM_EOS])
            posterior.append(vocab.word2idx[vocab.SYM_PAD])
        except:
            pass

        answer = self.answers[index].tolist()
        try:
            answer.remove(vocab.word2idx[vocab.SYM_EOS])
            answer.append(vocab.word2idx[vocab.SYM_PAD])
        except:
            answer = answer

        # gives us an index of sorted cat2name
        answer_type = self.answer_types[index]
        answer_type = vocab.word2idx[self.cat2name[answer_type]]

        answer_type_for_input = [
            vocab.word2idx[vocab.SYM_SOQ], answer_type, vocab.word2idx[vocab.SYM_EOS]]
        answer_type_for_input = torch.from_numpy(
            np.array(answer_type_for_input))

        posterior.insert(1, answer_type)
        posterior = np.array(posterior)

        ##############################
        # sym_pad = vocab.word2idx[vocab.SYM_PAD]

        # question_contents = copy.deepcopy(question.tolist())
        # try:
        #     question_contents.remove(vocab.word2idx[vocab.SYM_EOS])
        # except:
        #     pass
        # while sym_pad in question_contents:
        #     question_contents.remove(sym_pad)

        # answer_contents = copy.deepcopy(answer)
        # try:
        #     answer_contents.remove(vocab.word2idx[vocab.SYM_EOS])
        # except:
        #     pass
        # while sym_pad in answer_contents:
        #     answer_contents.remove(sym_pad)

        # posterior = [vocab.word2idx[vocab.SYM_POS], answer_type, *question_contents, *answer_contents, vocab.word2idx[vocab.SYM_EOS]]
        # while len(posterior) < self.max_len:
        #     posterior.append(sym_pad)

        # if len(posterior) > self.max_len: posterior = posterior[:self.max_len]
        # posterior = np.array(posterior)
        # print(posterior)
        ################################

        answer.insert(1, answer_type)
        answer = np.array(answer)

        image_index = self.image_indices[index]
        image = self.images[image_index]
        image_id = self.image_ids[index]

        question = torch.from_numpy(question)

        posterior = torch.from_numpy(posterior)
        answer = torch.from_numpy(answer)
        alength = answer.size(0) - answer.eq(0).sum(0).squeeze()
        qlength = question.size(0) - question.eq(0).sum(0).squeeze()
        if self.transform is not None:
            image = self.transform(image)
        return (image, image_id, question, posterior, answer, answer_type, answer_type_for_input,
                qlength.item(), alength.item())

    def __len__(self):
        if self.max_examples is not None:
            return self.max_examples
        if self.indices is not None:
            return len(self.indices)
        annos = h5py.File(self.dataset, 'r')
        return annos['questions'].shape[0]


def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples.

    We should build custom collate_fn rather than using default collate_fn,
    because merging caption (including padding) is not supported in default.

    Args:
        data: list of tuple (image, question, answer, answer_type, length).
            - image: torch tensor of shape (3, 256, 256).
            - question: torch tensor of shape (?); variable length.
            - answer: torch tensor of shape (?); variable length.
            - answer_type: Int for category label
            - qlength: Int for question length.
            - alength: Int for answer length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        questions: torch tensor of shape (batch_size, padded_length).
        answers: torch tensor of shape (batch_size, padded_length).
        answer_types: torch tensor of shape (batch_size,).
        qindices: torch tensor of shape(batch_size,).
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: x[5], reverse=True)
    images, image_ids, questions, posteriors, answers, answer_types, answer_types_for_input, qlengths, _ = list(
        zip(*data))
    images = torch.stack(images, 0)
    questions = torch.stack(questions, 0).long()
    posteriors = torch.stack(posteriors, 0).long()
    answers = torch.stack(answers, 0).long()
    answer_types = torch.Tensor(answer_types).long()
    answer_types_for_input = torch.stack(answer_types_for_input, 0).long()
    qindices = np.flip(np.argsort(qlengths), axis=0).copy()
    qindices = torch.Tensor(qindices).long()
    return {"images": images, "image_ids": image_ids, "questions": questions, "posteriors": posteriors, "answers": answers, "answer_types": answer_types, "answer_types_for_input": answer_types_for_input, "qindicies": qindices}


def get_loader(dataset, transform, batch_size, sampler=None,
               shuffle=True, num_workers=1, max_examples=None,
               indices=None):
    """Returns torch.utils.data.DataLoader for custom dataset.

    Args:
        dataset: Location of annotations hdf5 file.
        transform: Transformations that should be applied to the images.
        batch_size: How many data points per batch.
        sampler: Instance of WeightedRandomSampler.
        shuffle: Boolean that decides if the data should be returned in a
            random order.
        num_workers: Number of threads to use.
        max_examples: Used for debugging. Assumes that we have a
            maximum number of training examples.
        indices: List of indices to use.

    Returns:
        A torch.utils.data.DataLoader for custom engagement dataset.
    """
    iq = IQDataset(dataset, transform=transform, max_examples=max_examples,
                   indices=indices)
    data_loader = torch.utils.data.DataLoader(dataset=iq,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              sampler=sampler,
                                              num_workers=num_workers,
                                              collate_fn=collate_fn)
    return data_loader
