var express = require('express');
var multer  = require('multer');
const { exec } = require("child_process");
var path = require('path');

var fs  = require('fs');

var app = express();
app.set('view engine', 'ejs');

app.get('/', (req, res) => {
    res.render('index');
});

var storage = multer.diskStorage({
    destination: function (req, file, callback) {
        var dir = './uploads';
        if (!fs.existsSync(dir)){
            fs.mkdirSync(dir);
        }
        callback(null, dir);
    },
    filename: function (req, file, callback) {
        callback(null, file.originalname);
    }
});
var upload = multer({storage: storage}).array('files', 12);
app.post('/upload', function (req, res, next) {
    upload(req, res, function (err) {
        if (err) {
            return res.end("Something went wrong:(");
        }
                exec("python python.py", (error, stdout, stderr) => {
            if (error) {
                console.log(`error: ${error.message}`);
                       }
                }
                 res.sendFile('somefile.txt', options, function (err) {
                if (err) {
                    next(err);
                } else {
                    console.log("Sent:");
                }
            });
    });
})

app.listen(process.env.PORT || 3000);
