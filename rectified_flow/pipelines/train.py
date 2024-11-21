def train(self, num_iterations=100, num_epochs=None, batch_size=64, D0=None, D1=None, optimizer=None, shuffle=True):
        # Will be deprecated!!!
        if optimizer is None: optimizer = self.optimizer
        if num_iterations is None: num_iterations = self.num_iterations
        if num_epochs is not None: num_epochs = self.num_epochs
        self.loss_curve = []

        # Set up dataloader
        dataloader = torch.utils.data.DataLoader(self.dataset, batch_size=batch_size, shuffle=shuffle)
        self.dataloader = dataloader

        # Calculate the total number of batches to process
        total_batches = num_iterations
        if num_epochs is not None:
            total_batches = num_epochs * len(dataloader)

        batch_count = 0
        while batch_count < total_batches:
            for batch in dataloader: # batch can be either (X0,X1) or (X0,X1,labels)

                if batch_count >= total_batches: break
                optimizer.zero_grad()

                # loss & backprop
                loss = self.get_loss(*batch)
                loss.backward()
                optimizer.step()

                # Track loss
                self.loss_curve.append(loss.item())
                batch_count += 1

 def set_random_seed(self, seed=None):
        # Will be deprecated!!!
        if seed is None: seed = self.seed
        self.seed = seed  
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # For deterministic behavior in certain operations (may affect performance)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def plot_loss_curve(self):
          plt.plot(self.loss_curve, '-.')