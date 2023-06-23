from datasets import Dataset


class AcronymDataset:
    def __init__(self, file_path):
        self._file_path = file_path
        self._dataset = None

        self._create_dataset()
    
    @property
    def data(self):
        return self._dataset
    
    def _create_dataset(self):
        data = []

        with open(self._file_path, "r", errors='ignore') as file:
            for line in file.readlines():
                split = line.strip().split('|')
                
                # build the sentence structure
                source_sentence = split[6]
                full_name = split[1]
                compare_sentence = source_sentence[:int(split[3])] + full_name + source_sentence[int(split[4]) + 1:]

                row = {
                    'source_sentence': source_sentence,
                    'compare_sentence': compare_sentence,
                    'label': 1,
                    'acronym': split[0]
                }
                data.append(row)

        data_dict = {key: [item[key] for item in data] for key in data[0]}
        self._dataset = Dataset.from_dict(data_dict)

    def _create_negative_examples(self):
        pass