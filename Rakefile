require 'rake/clean'

SOURCE_FILES = Rake::FileList.new('**/*.py') do |fl|
  fl.exclude(/^scratch\//)
  fl.exclude do |f|
    `git ls-files #{f}`.empty?
  end
end

PLOTS = Rake::FileList.new('**/*.png')
SERIALIZATION = Rake::FileList.new('**/*.pkl')
DATASETS = Rake::FileList.new('data/mnist.pkl.gz, data/cifar-10-python.tar')

CLEAN << [ SERIALIZATION, DATASETS ]
CLOBBER << PLOTS

namespace :logreg do
  file 'logreg/best_model.pkl' do
    %x(python logreg/logistic_regression.py train adam 1>&2)
  end

  file 'logreg/repflds.png' => 'logreg/best_model.pkl' do
    %x(python logreg/logistic_regression.py plot repflds 1>&2)
  end

  file 'logreg/error.png' => 'logreg/best_model.pkl' do
    %x(python logreg/logistic_regression.py plot error 1>&2)
  end

  task :predict => 'logreg/best_model.pkl' do
    %x(python logreg/logistic_regression.py predict 1>&2)
  end
end

task logreg: ['logreg/repflds.png', 'logreg/error.png', 'logreg:predict']

namespace :nn do

  task :predict => 'logreg/best_model.pkl' do
    %x(python nn/neural_net.py predict 1>&2)
  end

  namespace :tanh do

    file 'nn/error_tanh.png' do
      %x(python nn/neural_net.py tanh gd 1>&2)
    end

    file 'nn/repflds_tanh.png' do
      %x(python nn/neural_net.py tanh gd 1>&2)
    end
  end
  task tanh: ['nn/repflds_tanh.png', 'nn/error_tanh.png']

  namespace :sigmoid do
    file 'nn/error_sigmoid.png' do
      %x(python nn/neural_net.py sigmoid gd 1>&2)
    end

    file 'nn/repflds_sigmoid.png' do
      %x(python nn/neural_net.py sigmoid gd 1>&2)
    end
  end
  task sigmoid: ['nn/repflds_sigmoid.png', 'nn/error_sigmoid.png']

  namespace :relu do
    file 'nn/error_relu.png' do
      %x(python nn/neural_net.py relu gd 1>&2)
    end

    file 'nn/repflds_relu.png' do
      %x(python nn/neural_net.py relu gd 1>&2)
    end
  end
  task relu: ['nn/repflds_relu.png', 'nn/error_relu.png']
end
task nn: ['nn:tanh', 'nn:sigmoid', 'nn:relu']

namespace :tsne do
  file 'tsne/bh_tsne' do
    %x(curl https://lvdmaaten.github.io/tsne/code/bh_tsne.tar.gz | tar xz -C tsne)
  end

  file 'tsne/bh_tsne/bh_tsne' => 'tsne/bh_tsne' do
    %x(g++ tsne/bh_tsne/sptree.cpp tsne/bh_tsne/tsne.cpp -o tsne/bh_tsne/bh_tsne -O2)
  end

  file 'tsne/data.pkl' => 'tsne/bh_tsne/bh_tsne' do
    %x(python tsne/tsne_mnist.py train 1>&2)
  end

  file 'tsne/tsne_mnist.png' do
    %x(python tsne/tsne_mnist.py plot 1>&2)
  end
end
task :tsne ['tsne/tsne_mnist.png']

namespace :latent do
  namespace :pca do
      file 'latent/scatterplotMNIST.png' do
        %x(python latent/pca.py 1>&2)
      end

    file 'latent/scatterplotCIFAR.png' do
      %x(python latent/pca.py 1>&2)
    end
  end
  task pca: ['latent/scatterplotMNIST.png', 'latent/scatterplotCIFAR.png']

  namespace :autoencoder do
    file 'latent/autoencoder.pkl' do
      %x(python latent/dA.py train 1>&2)
    end

    file 'latent/autoencoderrec.png' => 'latent/autoencoder.pkl' do
      %x(python latent/dA.py plot reconstructions 1>&2)
    end

    file 'latent/autoencoderfilter.png' => 'latent/autoencoder.pkl' do
      %x(python latent/dA.py plot repflds 1>&2)
    end
  end
  task autoencoder: ['latent/autoencoderrec.png', 'latent/autoencoderfilter.png']
end
task latent: ['latent:pca', 'latent:autoencoder']

namespace :kmeans do
  file 'kmeans/best_model.pkl' do
    %x(python kmeans/kmeans.py train 1>&2)
  end

  file 'kmeans/repflds.png' => 'kmeans/best_model.pkl' do
    %x(python kmeans/kmeans.py plot 1>&2)
  end
end

task kmeans: 'kmeans/repflds.png'


task default: [:logreg, :nn, :latent, :tsne, :kmeans]
