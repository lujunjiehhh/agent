import React, { useEffect, useState } from 'react';
import { Card, Modal, List, Typography, Spin } from 'antd';
import * as d3 from 'd3';
import axios from 'axios';

const { Title, Paragraph } = Typography;

interface Entity {
  id: string;
  name: string;
  type: string;
  properties?: any;
}

interface Relation {
  source_id: string;
  target_id: string;
  type: string;
  properties?: any;
}

interface KnowledgeGraph {
  entities: Entity[];
  relations: Relation[];
}

interface EntityRelation {
  entity: Entity;
  related_documents: any[];
  relations: Relation[];
}

const KnowledgeGraph: React.FC = () => {
  const [graph, setGraph] = useState<KnowledgeGraph | null>(null);
  const [loading, setLoading] = useState(true);
  const [selectedEntity, setSelectedEntity] = useState<Entity | null>(null);
  const [entityDetails, setEntityDetails] = useState<EntityRelation | null>(null);
  const [modalVisible, setModalVisible] = useState(false);

  useEffect(() => {
    fetchGraph();
  }, []);

  useEffect(() => {
    if (graph) {
      renderGraph();
    }
  }, [graph]);

  const fetchGraph = async () => {
    try {
      const response = await axios.get('http://localhost:8000/api/knowledge-graph');
      setGraph(response.data);
    } catch (error) {
      console.error('Error fetching graph:', error);
    } finally {
      setLoading(false);
    }
  };

  const fetchEntityDetails = async (entityId: string) => {
    try {
      const response = await axios.get(`http://localhost:8000/api/knowledge-graph/entities/${entityId}/related`);
      setEntityDetails(response.data);
      setModalVisible(true);
    } catch (error) {
      console.error('Error fetching entity details:', error);
    }
  };

  const renderGraph = () => {
    if (!graph) return;

    // 清除现有的SVG
    d3.select('#graph-container').selectAll('*').remove();

    const width = 800;
    const height = 600;
    const margin = { top: 20, right: 20, bottom: 20, left: 20 };

    const svg = d3.select('#graph-container')
      .append('svg')
      .attr('width', width)
      .attr('height', height);

    // 创建力导向图
    const simulation = d3.forceSimulation(graph.entities)
      .force('link', d3.forceLink(graph.relations)
        .id((d: any) => d.id)
        .distance(100))
      .force('charge', d3.forceManyBody().strength(-300))
      .force('center', d3.forceCenter(width / 2, height / 2));

    // 绘制关系线
    const links = svg.append('g')
      .selectAll('line')
      .data(graph.relations)
      .enter()
      .append('line')
      .attr('stroke', '#999')
      .attr('stroke-width', 1);

    // 创建节点组
    const nodes = svg.append('g')
      .selectAll('g')
      .data(graph.entities)
      .enter()
      .append('g')
      .call(d3.drag()
        .on('start', dragstarted)
        .on('drag', dragged)
        .on('end', dragended));

    // 添加节点圆圈
    nodes.append('circle')
      .attr('r', 20)
      .attr('fill', '#69b3a2')
      .attr('cursor', 'pointer')
      .on('click', (event, d: any) => {
        setSelectedEntity(d);
        fetchEntityDetails(d.id);
      });

    // 添加节点文本
    nodes.append('text')
      .text((d: any) => d.name)
      .attr('text-anchor', 'middle')
      .attr('dy', 5)
      .attr('fill', 'white')
      .style('font-size', '12px');

    // 更新力导向图
    simulation.on('tick', () => {
      links
        .attr('x1', (d: any) => d.source.x)
        .attr('y1', (d: any) => d.source.y)
        .attr('x2', (d: any) => d.target.x)
        .attr('y2', (d: any) => d.target.y);

      nodes
        .attr('transform', (d: any) => `translate(${d.x},${d.y})`);
    });

    // 拖拽函数
    function dragstarted(event: any) {
      if (!event.active) simulation.alphaTarget(0.3).restart();
      event.subject.fx = event.subject.x;
      event.subject.fy = event.subject.y;
    }

    function dragged(event: any) {
      event.subject.fx = event.x;
      event.subject.fy = event.y;
    }

    function dragended(event: any) {
      if (!event.active) simulation.alphaTarget(0);
      event.subject.fx = null;
      event.subject.fy = null;
    }
  };

  return (
    <div>
      <Card title="知识图谱">
        <Spin spinning={loading}>
          <div id="graph-container" style={{ width: '100%', height: '600px' }} />
        </Spin>
      </Card>

      <Modal
        title={selectedEntity?.name}
        open={modalVisible}
        onCancel={() => setModalVisible(false)}
        width={800}
        footer={null}
      >
        {entityDetails && (
          <div>
            <Title level={4}>实体信息</Title>
            <Paragraph>
              类型: {entityDetails.entity.type}
            </Paragraph>
            {entityDetails.entity.properties && (
              <Paragraph>
                属性: {JSON.stringify(entityDetails.entity.properties, null, 2)}
              </Paragraph>
            )}

            <Title level={4}>相关文档</Title>
            <List
              dataSource={entityDetails.related_documents}
              renderItem={(doc) => (
                <List.Item>
                  <Card size="small" style={{ width: '100%' }}>
                    <Paragraph>{doc.filename}</Paragraph>
                  </Card>
                </List.Item>
              )}
            />

            <Title level={4}>关系</Title>
            <List
              dataSource={entityDetails.relations}
              renderItem={(relation) => (
                <List.Item>
                  <Card size="small" style={{ width: '100%' }}>
                    <Paragraph>
                      类型: {relation.type}
                      {relation.properties && ` (${JSON.stringify(relation.properties)})`}
                    </Paragraph>
                  </Card>
                </List.Item>
              )}
            />
          </div>
        )}
      </Modal>
    </div>
  );
};

export default KnowledgeGraph; 