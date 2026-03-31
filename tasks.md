don't commit any changes, i have this file @modal_unav_v2.py , there is a .venv which you can activate
you can server it using modal server {file_name}
when deploying i got this, issue, i'll paste log
you job is fix it, run fixes, server the app, hit the url, if there is any issue automatically fix it, then test your changes, once the app is deployed successfully hit the url of fastapi app , if you curl the root url, you can watch for modal fast api logs since you'll be using serve command, based on the error logs , automatically fix the error and retry again

keep doing this untill everything works, stop only if there is anny issue with data_root else if depencendy or anytother issue keep fixing in loop 

pulling the image and building the modal image can take about 20 minues or more pls keep it mind something close to 30 mins

you can't build the image locally because of my machine, but you can push the fix to github and it would deploy the image on every push 