require "torch"
require "json"


data = json.load("../train.json")

-- Generate the bitmaps for each cuisine type
-- Generate list of possible ingredients
labels = {}
ingredients = {}
label_ind = 0
ingred_ind = 0
for _, v in pairs(data) do
    if v["cuisine"] ~= nil and labels[v["cuisine"]] == nil then
        label_ind = label_ind + 1
        labels[v["cuisine"]] = label_ind
    end

    for _, ingredient in pairs(v["ingredients"]) do
        if ingredients[ingredient] == nil then
            ingred_ind = ingred_ind + 1
            ingredients[ingredient] = ingred_ind
        end
    end
end
if label_ind ~= 0 then
    torch.save("label-mappings.txt", labels)
end

-- Generate the bitmaps that will be used in place of the label
labelmap = {}
for k, v in pairs(labels) do 
    map = {}
    for i = 1, label_ind do
        if i == v then
            table.insert(map, 1)
        else
            table.insert(map, 0)
        end
    end
    labelmap[k] = map
end

-- Generate new data set using the appropriate bitmaps
for _, v in pairs(data) do
    sample = {}

    for ingredient, index in pairs(ingredients) do
        for _, ing in pairs(v["ingredients"]) do
            if ing == ingredient then
                sample[index] = 1
                break
            end
            sample[index] = 0
        end
    end
    
    for i = 1,#sample do 
        if sample[i] ~= 0 then
            io.write(i  .. ":" .. 1 .. " ")
        end
    end
    io.write("\n")
end
