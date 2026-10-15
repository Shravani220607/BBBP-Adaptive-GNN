
class Trainer:
    def __init__(self, model, optimizer, criterion, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0.0

        for batch in dataloader:
            graph = batch["graph"].to(self.device)
            line_graph = batch["line_graph"].to(self.device)

            distances = batch.get("distance_graphs", batch["distance_graph"])

            if isinstance(distances, (list, tuple)):
                distances = [graph.to(self.device) for graph in distances]
            else:
                distances = distances.to(self.device)

            descriptors = batch["descriptor"].to(self.device)
            geometry = batch["geometry"]
            labels = batch["label"].to(self.device)

            if geometry is not None:
                geometry = geometry.to(self.device)

            self.optimizer.zero_grad()

            predictions = self.model(
                graph_data=graph,
                line_data=line_graph,
                distance_data=distances,
                descriptors=descriptors,
                geometry_data=geometry,
            )

            loss = self.criterion(predictions.view(-1), labels.view(-1))

            if hasattr(self.model, "get_regularization_loss"):
                loss = loss + self.model.get_regularization_loss()

            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()

        if len(dataloader) == 0:
            raise ValueError("The training data loader is empty.")

        return total_loss / len(dataloader)
