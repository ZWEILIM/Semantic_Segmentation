class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == '7b52009b64fd0a2a49e6d8a939753077792b0554':
            # return r'E:\OneDrive - mmu.edu.my\Documents - ITP\lzw\tia_coanet\data\7b52009b64fd0a2a49e6d8a939753077792b0554'  # Update this with the actual root path of your dataset
            return r'C:\Program Files (x86)\Study\Degree\Intern\Intern Project\CoANet\data\7b52009b64fd0a2a49e6d8a939753077792b0554'  # Update this with the actual root path of your dataset
        
        elif dataset == '0ade7c2cf97f75d009975f4d720d1fa6c19f4897':
            return r'E:\OneDrive - mmu.edu.my\Documents - ITP\lzw\tia_coanet\data\0ade7c2cf97f75d009975f4d720d1fa6c19f4897'  # Update this with the actual root path of your dataset
            # return r'C:\Program Files (x86)\Study\Degree\Intern\Intern Project\CoANet\data\0ade7c2cf97f75d009975f4d720d1fa6c19f4897'  # Update this with the actual root path of your dataset
       
        elif dataset == '1b6453892473a467d07372d45eb05abc2031647a':
            return r'E:\OneDrive - mmu.edu.my\Documents - ITP\lzw\tia_coanet\data\1b6453892473a467d07372d45eb05abc2031647a'  # Update this with the actual root path of your dataset
            # return r'C:\Program Files (x86)\Study\Degree\Intern\Intern Project\CoANet\data\1b6453892473a467d07372d45eb05abc2031647a'  # Update this with the actual root path of your dataset
        
        elif dataset == '17ba0791499db908433b80f37c5fbc89b870084b':
            return r'E:\OneDrive - mmu.edu.my\Documents - ITP\lzw\tia_coanet\data\17ba0791499db908433b80f37c5fbc89b870084b'  # Update this with the actual root path of your dataset
            # return r'C:\Program Files (x86)\Study\Degree\Intern\Intern Project\CoANet\data\17ba0791499db908433b80f37c5fbc89b870084b'  # Update this with the actual root path of your dataset
        
        elif dataset == '356a192b7913b04c54574d18c28d46e6395428ab':
            return r'E:\OneDrive - mmu.edu.my\Documents - ITP\lzw\tia_coanet\data\356a192b7913b04c54574d18c28d46e6395428ab'  # Update this with the actual root path of your dataset
            # return r'C:\Program Files (x86)\Study\Degree\Intern\Intern Project\CoANet\data\356a192b7913b04c54574d18c28d46e6395428ab'  # Update this with the actual root path of your dataset
        
        elif dataset == '902ba3cda1883801594b6e1b452790cc53948fda':
            return r'E:\OneDrive - mmu.edu.my\Documents - ITP\lzw\tia_coanet\data\902ba3cda1883801594b6e1b452790cc53948fda'  # Update this with the actual root path of your dataset
            # return r'C:\Program Files (x86)\Study\Degree\Intern\Intern Project\CoANet\data\902ba3cda1883801594b6e1b452790cc53948fda'  # Update this with the actual root path of your dataset
        
        elif dataset == 'b1d5781111d84f7b3fe45a0852e59758cd7a87e5':
            return r'E:\OneDrive - mmu.edu.my\Documents - ITP\lzw\tia_coanet\data\b1d5781111d84f7b3fe45a0852e59758cd7a87e5'  # Update this with the actual root path of your dataset
            # return r'C:\Program Files (x86)\Study\Degree\Intern\Intern Project\CoANet\data\b1d5781111d84f7b3fe45a0852e59758cd7a87e5'  # Update this with the actual root path of your dataset
        
        elif dataset == 'bd307a3ec329e10a2cff8fb87480823da114f8f4':
            return r'E:\OneDrive - mmu.edu.my\Documents - ITP\lzw\tia_coanet\data\bd307a3ec329e10a2cff8fb87480823da114f8f4'  # Update this with the actual root path of your dataset
            # return r'C:\Program Files (x86)\Study\Degree\Intern\Intern Project\CoANet\data\bd307a3ec329e10a2cff8fb87480823da114f8f4'  # Update this with the actual root path of your dataset
        
        elif dataset == 'c1dfd96eea8cc2b62785275bca38ac261256e278':
            return r'E:\OneDrive - mmu.edu.my\Documents - ITP\lzw\tia_coanet\data\c1dfd96eea8cc2b62785275bca38ac261256e278'  # Update this with the actual root path of your dataset
            # return r'C:\Program Files (x86)\Study\Degree\Intern\Intern Project\CoANet\data\c1dfd96eea8cc2b62785275bca38ac261256e278'  # Update this with the actual root path of your dataset
        
        elif dataset == 'fa35e192121eabf3dabf9f5ea6abdbcbc107ac3b':
            return r'E:\OneDrive - mmu.edu.my\Documents - ITP\lzw\tia_coanet\data\fa35e192121eabf3dabf9f5ea6abdbcbc107ac3b'  # Update this with the actual root path of your dataset
            # return r'C:\Program Files (x86)\Study\Degree\Intern\Intern Project\CoANet\data\fa35e192121eabf3dabf9f5ea6abdbcbc107ac3b'  # Update this with the actual root path of your dataset
        
        elif dataset == 'mergedataset':
            return r'E:\OneDrive - mmu.edu.my\Documents - ITP\lzw\tia_coanet\data\merged_dataset'  # Update this with the actual root path of your dataset
            # return r'C:\Program Files (x86)\Study\Degree\Intern\Intern Project\CoANet\data\merged_dataset'  # Update this with the actual root path of your dataset
        
        else:
            print('Dataset not recognized: {}'.format(dataset))
            return ''
