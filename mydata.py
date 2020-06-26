# import torch
# from torch_geometric.data import InMemoryDataset


# class FcadDataset(InMemoryDataset):
#     def __init__(self, root_dir, dtype='train', transform=None, pre_transform=None):
#         self.root_dir = root_dir
#         self.data_file = glob.glob(self.root_dir + '/*/*/*.ply')

#         super(FcadDataset, self).__init__(root, transform, pre_transform)

#         if dtype == 'train':
#             data_path = self.processed_paths[0]
#         elif dtype == 'val':
#             data_path = self.processed_paths[1]
#         elif dtype == 'test':
#             data_path = self.processed_paths[2]
#         else:
#             raise Exception("train, val and test are supported data types")

#         # print(self.data_file)
#         norm_path = self.processed_paths[3]
#         norm_dict = torch.load(norm_path)
#         self.mean, self.std = norm_dict['mean'], norm_dict['std']
#         self.data, self.slices = torch.load(data_path)

#     @property
#     def raw_file_names(self):
#         return self.data_file

#     @property
#     def processed_file_names(self):
#         processed_files = ['training.pt', 'val.pt', 'test.pt', 'norm.pt']
#         processed_files = [self.split_term+'_'+pf for pf in processed_files]
#         return processed_files

#     # def download(self):
#     #     # Download to `self.raw_dir`.

#     def process(self):
#         # Read data into huge `Data` list.
#         data_list = [...]

#         data.x = data.x - data.x.mean(dim=-2, keepdim=True)
#         scale = (1 / data.x.abs().max()) * 0.999999
#         data.x = data.x * scale


#         if self.pre_filter is not None:
#             data_list = [data for data in data_list if self.pre_filter(data)]

#         if self.pre_transform is not None:
#             data_list = [self.pre_transform(data) for data in data_list]

#         data, slices = self.collate(data_list)
#         torch.save((data, slices), self.processed_paths[0])