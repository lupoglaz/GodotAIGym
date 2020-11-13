function generate_row_header(array){
    var row = document.createElement("tr");
    for(var i=0; i<4; i++){
        var cell = document.createElement("th");
        cell.setAttribute("scope", "col");
        var cell_text = document.createTextNode(array[i]);
        cell.appendChild(cell_text);
        row.append(cell);
    }
    return row;
}
function generate_row(array){
    var row = document.createElement("tr");
    for(var i=0; i<4; i++){
        var cell;
        if(i==0){
            cell = document.createElement("th");
            cell.setAttribute("scope", "row");
        }else{
            cell = document.createElement("td");
        }
        
        var cell_text = document.createTextNode(array[i]);
        cell.appendChild(cell_text);
        row.append(cell);
    }
    return row;
}

function generate_table(container,
                        inputs,
                        outputs,
                        parameters = []
                        ){
    
    var container = document.getElementById(container);
    
    var tbl = document.createElement("table");
    tbl.className = "table table-sm";

    var tblThead = document.createElement("thead");
    tblThead.className = "thead-light";
    var header_row = generate_row_header(["Name", "Device", "Type", "Size"]);
    tblThead.appendChild(header_row);
    tbl.appendChild(tblThead);

    var tblHInputs = document.createElement("thead");
    tblHInputs.className = "thead-light";
    var header_inputs = generate_row_header(["Inputs", "", "", ""]);
    tblHInputs.appendChild(header_inputs);
    tbl.appendChild(tblHInputs);
    
    var tblInBody = document.createElement("tbody");
    for(var i=0; i<inputs.length; i++){
        var row = generate_row([inputs[i][0], inputs[i][1], inputs[i][2], inputs[i][3]]);
        tblInBody.appendChild(row);
    }
    tbl.appendChild(tblInBody);
    
    var tblHOutputs = document.createElement("thead");
    tblHOutputs.className = "thead-light";
    var header_outputs = generate_row_header(["Outputs", "", "", ""]);
    tblHOutputs.appendChild(header_outputs);
    tbl.appendChild(tblHOutputs);
    
    var tblOutBody = document.createElement("tbody");
    for(var i=0; i<outputs.length; i++){
        var row = generate_row([outputs[i][0], outputs[i][1], outputs[i][2], outputs[i][3]]);
        tblOutBody.appendChild(row);
    }
    tbl.appendChild(tblOutBody);

    if(parameters.length>0){
        var tblHParams = document.createElement("thead");
        tblHParams.className = "thead-light";
        var header_outputs = generate_row_header(["Parameters", "", "", ""]);
        tblHParams.appendChild(header_outputs);
        tbl.appendChild(tblHParams);
        
        var tblParBody = document.createElement("tbody");
        for(var i=0; i<parameters.length; i++){
            var row = generate_row([parameters[i][0], parameters[i][1], parameters[i][2], parameters[i][3]]);
            tblParBody.appendChild(row);
        }
        tbl.appendChild(tblParBody);
    }

    container.appendChild(tbl);
}

function createHeader(container, type, module, name, params=""){
    var obj = document.getElementById(container);
    var obj_container = document.createElement("div");
    obj_container.className = "container";

    var header_obj = document.createElement("div");
    header_obj.className = "p-1 mb-4 bg-dark text-white";
    var text_obj = document.createElement("P");
    text_obj.className = "h4";
    
    var type_obj = document.createElement("EM");
    var type_text = document.createTextNode(type+' ');
    type_obj.appendChild(type_text);
    
    var module_obj = document.createElement('SMALL');
    module_obj.className = "text-muted";
    var module_text = document.createTextNode(module+'.');
    module_obj.appendChild(module_text);
    
    var class_obj = document.createElement('B');
    var class_text = document.createTextNode(name);
    class_obj.appendChild(class_text);

    header_obj.appendChild(type_obj);
    header_obj.appendChild(module_obj);
    header_obj.appendChild(class_obj);

    if(params.length>0){
        var params_text = document.createTextNode(params);
        header_obj.appendChild(params_text);    
    }
    
    text_obj.appendChild(header_obj);
    obj_container.appendChild(text_obj);
    obj.appendChild(obj_container);
}