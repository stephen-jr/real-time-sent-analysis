function eventSource(){
    const a = new EventSource('http://localhost:5000/test')
    a.onmessage = (stream) => {
        let { data } = stream;
        data = JSON.parse(data)
        console.log(data)
//        if(data.execution)console.log(data)
         if(!data.execution)a.close()
    }
}
setTimeout(eventSource, 3000)