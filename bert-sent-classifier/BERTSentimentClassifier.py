from absl import app, flags, logging
import sh 

import torch as th
import pytorch_lightning as pl

import nlp
import transformers

flags.DEFINE_bool("debug", False, "")
flags.DEFINE_integer("batch_size", 8, "") # batch size for debug

flags.DEFINE_string("model", "bert-base-uncased", "")
flags.DEFINE_integer("epochs", 10, "number of epochs")
flags.DEFINE_float("lr", 1e-2, "")
flags.DEFINE_float("momentum", 0.9, "")
flags.DEFINE_integer("seq_len", 32, "")
flags.DEFINE_integer("percent", 5, "")

FLAGS = flags.FLAGS

sh.rm("-r", "-f", "logs")
sh.mkdir("logs")


class IMDBSentimentClassifier(pl.LightningModule):
    def __init__(self):
        # setup BERT model from Transformers Huggingface library
        super().__init__()
        self.model = transformers.BertForSequenceClassification.from_pretrained(FLAGS.model)
        self.loss = th.nn.CrossEntropyLoss(reduction="none")

    def prepare_dataset(self):
        # train_data = nlp.load_dataset("imdb", split="train[:5%]")
        tokenizer = transformers.BertTokenizer.from_pretrained(FLAGS.model)
        # import IPython; IPython.embed(); exit(1)

        def _tokenize(x):
            x['input_ids'] = tokenizer.batch_encode_plus(
                x["text"], 
                max_length=FLAGS.seq_len, 
                pad_to_max_length=True)["input_ids"]
            # tokenizer.tokenize(train_data[0]["text"]) # for tokenized word pieces
            # tokenizer.encode(train_data[0]["text"]) # for tokenized word encodings
            return x

        def _prepare_data(split):
            data = nlp.load_dataset(
                "imdb", 
                split=f"{split}[:{FLAGS.batch_size if FLAGS.debug else f'{FLAGS.percent}%'}]"
                )
            data = data.map(_tokenize)
            data.set_format(type="torch", columns=["input_ids", "label"]) # output torch tensors
            return data

        self.train_data, self.test_data = map(_prepare_data, ("train", "test"))
        import IPython; IPython.embed(); exit(1)

    
    def forward(self, batch):
        mask = (input_ids != 0).float()
        logits = self.model(input_ids, mask)
        return logits


    def training_step(self, batch, batch_index):
        # what to do with the data
        logits = self.forward(batch["input_ids"])
        loss = self.loss(logits, batch["label"]).mean()
        return {"loss": loss, "log": {"train_loss": loss}}


    def validation_step(self, batch, batch_index):
        logits = self.forward(batch["input_ids"])
        loss = self.loss(logits, batch["label"])
        acc = (logits.argmax(-1) == batch["label"]).float()
        return {"loss": loss, "acc": acc}


    def validation_epoch_end(self, outputs):
        loss = th.cat([o["loss"] for o in outputs], 0).mean()
        acc = th.cat([o['acc'] for o in outputs], 0).mean()
        out = {"val_loss": loss, "val_acc": acc}
        return {**out, "log": out}
        

    def train_dataloader(self):
        return th.utils.data.dataloader(
            self.train_data,
            batch_size=FLAGS.batch_size,
            drop_last=True,
            shuffle=True
        )


    def val_dataloader(self):
        return th.utils.data.dataloader(
            self.test_data,
            batch_size=FLAGS.batch_size,
            drop_last=False,
            shuffle=False
        )


    def configure_optimizers(self):
        return th.optim.SGD(
            self.parameters(),
            lr = FLAGS.lr,
            momentum=FLAGS.momentum
        )


def main(_):
    # logging.info("hello")
    model = IMDBSentimentClassifier()
    trainer = pl.Trainer(
        default_root_dir = "logs",
        gpus = (1 if th.cuda.is_available() else 0),
        max_epochs=FLAGS.epochs,
        fast_dev_run=FLAGS.debug,
        logger=pl.loggers.TensorBoardLogger("logs/", name="imdb", version=0)
    )

    trainer.fit(model)

if __name__ == "__main__":
    # import IPython; IPython.embed(); exit(1)
    app.run(main)

